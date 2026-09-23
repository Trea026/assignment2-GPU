##flash_forward
import torch
def flash_forward(Q,K,V,B_q,B_k,causal = False):
    #N=64, d=16, B_q=B_k=8
    prefix = Q.shape[:-2]
    N,d = Q.shape[-2],Q.shape[-1]
    
    q_n = N // B_q #需要向上取整
    kv_n = N // B_k #需要向上取整
    #全局初始化
    O = torch.empty(prefix+(N,d),device=Q.device,dtype=Q.dtype)
    L = torch.empty(prefix+(N,),device=Q.device,dtype=Q.dtype)

    #Q tile外循环，输出O，L
    for j in range(q_n):
        A = torch.zeros(prefix+(B_q,d),device=Q.device,dtype=Q.dtype)
        m =  torch.full(prefix+(B_q,),float('-inf'),device= A.device,dtype=A.dtype)
        l = torch.zeros(prefix+(B_q,),device= A.device,dtype=A.dtype)
        q = Q[...,j*B_q :(j+1)*B_q,:]
        for i in range(kv_n):
            #KV内循环，Qtile每个Q的对应l,m,A
            k = K[...,i*B_k :(i+1)*B_k,:]
            v = V[...,i*B_k :(i+1)*B_k,:]
            s_i = q @ k.transpose(-2,-1) / d ** 0.5  #prefix + （B_q,B_k）
            #causal mask
            if causal:
                r = j*B_q + torch.arange(B_q)
                c = i*B_k +torch.arange(B_k)

                #整块可见 不mask 
                if r[0] >= c[-1]:
                    pass
                #整块不可见 全mask
                elif r[-1] < c[0]:
                    continue
                #部分可见 分情况mask
                else:
                    mask = c[None,:] > r[:,None]
                    #s_i = q @ k.transpose(-2,-1) / d ** 0.5  #B_q,B_k
                    s_i.masked_fill_(mask,float('-inf'))

            m_old = m  #B_q
            m_new = torch.max(s_i,dim=-1).values  #B_q
            m = torch.max(m_new,m_old)  #B_q
            l_old = l #B_q

            l = torch.sum(torch.exp(s_i - m[...,None]),dim=-1) + l_old*torch.exp(m_old-m) #B_q
            A_old = A #B_q,d
            A = torch.exp(s_i - m[...,None]) @ v + A_old * torch.exp(m_old - m)[...,None]
        O_j = A / l[...,None] #B_q,d
        L_j = m +torch.log(l) #B_q
        O[...,j*B_q :(j+1)*B_q,:] = O_j
        L[...,j*B_q :(j+1)*B_q] = L_j

    return L,O
            

B,H,N,d = 2,3,64,16
Q,K,V = torch.randn(B,H,N,d),torch.randn(B,H,N,d),torch.randn(B,H,N,d)
#Q,K,V = torch.randn(N,d),torch.randn(N,d),torch.randn(N,d)
S = Q@K.transpose(-2,-1) /d**0.5
mask = torch.triu(torch.ones(N,N,dtype=torch.bool),diagonal=1)
S_c = S.masked_fill(mask, float('-inf'))
B_q,B_k = 4,16
L,O = flash_forward(Q=Q,K=K,V=V,B_q=B_q,B_k=B_k)
L1,O1 = flash_forward(Q=Q,K=K,V=V,B_q=B_q,B_k=B_k,causal = True)

O_test = torch.allclose(torch.softmax(S,dim=-1)@ V,O,atol=1e-5)
L_test = torch.allclose(torch.logsumexp(S,dim=-1),L,atol=1e-5)
O_test1 = torch.allclose(torch.softmax(S_c,dim=-1)@ V,O1,atol=1e-5)
L_test1 = torch.allclose(torch.logsumexp(S_c,dim=-1),L1,atol=1e-5)
print("O",O_test)
print("L",L_test)
print("O1",O_test1)
print("L1",L_test1)

ref = torch.softmax(S_c, dim=-1) @ V
bad = ~torch.isclose(ref, O1, atol=1e-5).all(dim=-1)   # [N] bool，哪些 query 错
print(bad.nonzero().flatten())