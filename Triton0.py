
import torch

import triton 
import triton.language as tl
#DEVICE = triton.runtime.active.get_active_torch_device()
#—— 底层kernel——
@triton.jit
def add_kernel(x_ptr,y_ptr,output_ptr,n_elements,BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets,mask=mask)
    y = tl.load(y_ptr + offsets,mask=mask)
    output = x + y 
    tl.store(output_ptr + offsets ,output,mask=mask)
#—— wrapper #———
def add(x:torch.Tensor, y:torch.Tensor):
    out = torch.empty_like(x)
    n = x.numel()
    block_size =64
    grid = (triton.cdiv(n,  block_size),)

    add_kernel[grid](x,y,out,n,BLOCK_SIZE=block_size)

    return out

if __name__ == "__main__":
    n = 100
    x = torch.randn(n, device='cuda')
    y = torch.randn(n, device='cuda')
    out = add(x, y)
    print(torch.allclose(out, x + y))
    print(out[-8:])
    print((x + y)[-8:])