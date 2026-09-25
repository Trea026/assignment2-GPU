import torch

import triton 
import triton.language as tl

#———底层kernel—— kernel里没有张量没有张量
@triton.jit
def add_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, #5个位置指针
               
               stride_qb,stride_qn,stride_qd,
               stride_kb,stride_kn,stride_kd,
               stride_vb,stride_vn,stride_vd,
               stride_ob,stride_on,stride_od,
               stride_lb,stride_ln,
               N,d:tl.constexpr,B_q:tl.constexpr,B_k:tl.constexpr,
               causal:tl.constexpr,
               scale):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    #block_start = pid_q * block_size
    rows = pid_q * B_q +tl.arange(0,B_q)
    cols = tl.arange(0, d)
    addr = Q_ptr + pid_bh * stride_qb + rows[:,None]*stride_qn + cols[None,:]*stride_qd
    mask = rows[:,None] < N
    q =  tl.load(addr,mask = mask)
    #内层循环
    m = tl.full((B_q,),float('-inf'),dtype=tl.float32)
    l = tl.zeros((B_q,),dtype=tl.float32)
    A = tl.zeros((B_q,d),dtype=tl.float32)
    for start_k in range(0,N,B_k):
        rows_k = start_k + tl.arange(0,B_k)
        cols_k = tl.arange(0,d)
        addr_k = K_ptr + pid_bh * stride_kb +rows_k[:,None]* stride_kn+cols_k[None,:]* stride_kd
        addr_v = V_ptr + pid_bh * stride_vb +rows_k[:,None]* stride_vn+cols_k[None,:]* stride_vd

        #mask_k = rows_k[:,None] > rows[:,None] 暂时不做
        k = tl.load(addr_k,mask = rows_k[:,None] < N, other = 0.0) 
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(rows_k[None,:]<N, s, float('-inf'))
        if causal:
            s = tl.where(rows[:,None] >= rows_k[None,:],s,float('-inf'))
        v = tl.load(addr_v,mask = rows_k[:,None]<N, other= 0.0,)

        m_new = tl.max(s,axis=1)
        m_old = m
        m = tl.maximum(m_new,m_old)
        l_old = l
        l = tl.sum(tl.exp(s - m[:,None]),axis=1) + l_old * tl.exp(m_old -m)
        A_old = A
        A = tl.dot(tl.exp(s - m[:,None]),v) + A_old * tl.exp(m_old - m)[:,None]

    o = A / l[:,None]
    addr_o = O_ptr + pid_bh * stride_ob + rows[:,None] *stride_on + cols[None,:]*stride_od
    tl.store(addr_o, o, mask = mask )
    addr_l = L_ptr + pid_bh * stride_lb + rows * stride_ln
    tl.store(addr_l, m + tl.log(l), mask = rows < N)



#——warpper—— 处理tensor，处理成kernel可处理的模样
def add(Q,K,V,B_q,B_k,causal=False):
    B,H,N,d = Q.shape
    O = torch.empty(Q.shape,device=Q.device,dtype=Q.dtype)
    L = torch.empty(Q.shape[0:-1],device=Q.device,dtype=Q.dtype)
    Q = Q.reshape(B*H, N, d)
    stride_qb,stride_qn,stride_qd = Q.stride(0),Q.stride(1),Q.stride(2)
    K = K.reshape(B*H, N, d)
    stride_kb,stride_kn,stride_kd = K.stride(0),K.stride(1),K.stride(2)
    V = V.reshape(B*H, N, d)
    stride_vb,stride_vn,stride_vd = V.stride(0),V.stride(1),V.stride(2)
    O = O.reshape(B*H, N, d)
    stride_ob,stride_on,stride_od = O.stride(0),O.stride(1),O.stride(2)
    L = L.reshape(B*H,N)
    stride_lb,stride_ln = L.stride(0),L.stride(1)
    scale = 1.0 / d**0.5
                                                                     
    grid = (triton.cdiv(N,B_q), B *H)
    add_kernel[grid](Q, K, V, O, L, #5个位置指针
               stride_qb,stride_qn,stride_qd,
               stride_kb,stride_kn,stride_kd,
               stride_vb,stride_vn,stride_vd,
               stride_ob,stride_on,stride_od,
               stride_lb,stride_ln,
               N,d,B_q,B_k,causal,scale)

    return O.reshape(B,H,N,d), L.reshape(B,H,N)

if __name__ == "__main__": 
    d = 16
    for causal in [False,True]:
        for (B_q, B_k) in [(16, 16), (32, 16), (16, 32)]:
            for N in [64, 100]:
                Q, K, V = [torch.randn(2, 3, N, 16, device='cuda') for _ in range(3)]
                S = Q @ K.transpose(-2, -1) / d**0.5
                if causal:
                    tri = torch.triu(torch.ones(N,N,dtype=torch.bool,device='cuda'),1)
                    S = S.masked_fill(tri,float('-inf'))
                ref_O = torch.softmax(S, -1) @ V
                ref_L = torch.logsumexp(S, -1)
                O, L = add(Q, K, V, B_q, B_k,causal=causal)
                bad = ~torch.isclose(ref_O, O, atol=1e-2).all(-1)
                print(bad[0, 0].nonzero().flatten())
    
                assert torch.allclose(ref_O, O, atol=1e-2), f"O,{B_q},{B_k},{N}"
                print("O", torch.allclose(ref_O, O, atol=1e-2))
                print("L", torch.allclose(ref_L, L, atol=1e-2))
                print((ref_O - O).abs().max().item(), (ref_L -L).abs().max().item())
                print(ref_L[0, 0, :4])
                print(L[0, 0, :4])

import triton.testing

B, H, N, d = 1, 8, 4096, 64
Q, K, V = [torch.randn(B, H, N, d, device='cuda') for _ in range(3)]
tri = torch.triu(torch.ones(N, N, dtype=torch.bool, device='cuda'), 1)

def ref():
    S = (Q @ K.transpose(-2, -1) / d**0.5).masked_fill(tri, float('-inf'))
    return torch.softmax(S, -1) @ V
    
torch.cuda.reset_peak_memory_stats()
ref()
torch.cuda.synchronize()
print("torch  peak:", torch.cuda.max_memory_allocated() / 1e9, "GB")


def mine():
    return add(Q, K, V, 64, 64, causal=True)

torch.cuda.reset_peak_memory_stats()
mine()
torch.cuda.synchronize()
print("triton peak:", torch.cuda.max_memory_allocated() / 1e9, "GB")

print("torch :", triton.testing.do_bench(ref), "ms")
print("triton:", triton.testing.do_bench(mine), "ms")





