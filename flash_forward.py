##flash_forward
import torch
def flash_forward(Q,K,V,B_q,B_k):
    #N=64, d=16, B_q=B_k=8
    N ,d = Q.shape[0],Q.shape[1]
    
    q_n = N // B_q #需要向上取整
    kv_n = N // B_k #需要向上取整
    #全局初始化
    O = torch.empty(N,d,device=Q.device,dtype=Q.dtype)
    L = torch.empty(N,device=Q.device,dtype=Q.dtype)

    #Q tile外循环，输出O，L
    for j in range(q_n):
        A = torch.zeros(B_q,d,device=Q.device,dtype=Q.dtype)
        m =  torch.full((B_q,),float('-inf'),device= A.device,dtype=A.dtype)
        l = torch.zeros(B_q,device= A.device,dtype=A.dtype)
        q = Q[j*B_q :(j+1)*B_q]
        for i in range(kv_n):
            #KV内循环，Qtile每个Q的对应l,m,A
            k = K[i*B_k :(i+1)*B_k]
            v = V[i*B_k :(i+1)*B_k]
            s_i = q @ k.T / d ** 0.5  #B_q,B_k
            m_old = m  #B_q
            m_new = torch.max(s_i,dim=-1).values  #B_q
            m = torch.max(m_new,m_old)  #B_q
            l_old = l #B_q
            l = torch.sum(torch.exp(s_i - m[:,None]),dim=-1) + l_old*torch.exp(m_old-m) #B_q
            A_old = A #B_q,d
            A = torch.exp(s_i - m[:,None]) @ v + A_old * torch.exp(m_old - m)[:,None]
        O_j = A / l[:,None] #B_q,d
        L_j = m +torch.log(l) #B_q
        O[j*B_q :(j+1)*B_q] = O_j
        L[j*B_q :(j+1)*B_q] = L_j

    return L,O
            

N,d = 64,16
Q,K,V = torch.randn(N,d),torch.randn(N,d),torch.randn(N,d)
S = Q@K.T /d**0.5
O_test = torch.allclose(torch.softmax(Q @ K.T /d**0.5,dim=-1)@ V,flash_forward(Q=Q,K=K,V=V,B_q=8,B_k=8)[1],atol=1e-5)
L_test = torch.allclose(torch.logsumexp(S,dim=-1),flash_forward(Q=Q,K=K,V=V,B_q=8,B_k=8)[0],atol=1e-5)
O_test1 = torch.allclose(torch.softmax(Q @ K.T /d**0.5,dim=-1)@ V,flash_forward(Q=Q,K=K,V=V,B_q=16,B_k=4)[1],atol=1e-5)
L_test1 = torch.allclose(torch.logsumexp(S,dim=-1),flash_forward(Q=Q,K=K,V=V,B_q=16,B_k=4)[0],atol=1e-5)
O_test2 = torch.allclose(torch.softmax(Q @ K.T /d**0.5,dim=-1)@ V,flash_forward(Q=Q,K=K,V=V,B_q=4,B_k=16)[1],atol=1e-5)
L_test2 = torch.allclose(torch.logsumexp(S,dim=-1),flash_forward(Q=Q,K=K,V=V,B_q=4,B_k=16)[0],atol=1e-5)
print("O",O_test)
print("L",L_test)
print("O1",O_test1)
print("L1",L_test1)
print("O2",O_test2)
print("L2",L_test2)