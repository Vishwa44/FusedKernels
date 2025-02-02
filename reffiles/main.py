import triton
import torch
from torch import nn
from typing import Optional, Tuple
import torch.nn.functional as F
import triton.language as tl
import time
# torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
torch.set_default_tensor_type(torch.cuda.HalfTensor)
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.feed_forward = FeedForward(dim = 4096, hidden_dim = 16384, multiple_of = 1024, ffn_dim_multiplier = 1.3)
        self.ffn_norm = RMSNorm(dim=4096, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        # start_pos: int,
        # freqs_cis: torch.Tensor,
        # mask: Optional[torch.Tensor],
    ):
        out = self.feed_forward(self.ffn_norm(x))
        return out


@triton.jit
def matmul_kernel(I, W1, W3, Out, RMS_W,
                  stride_ib, stride_im, stride_ik, 
                  stride_w1k, stride_w1n,
                  stride_w3k, stride_w3n,
                  stride_ob, stride_om, stride_on,
                  stride_rk,
                  B: tl.constexpr, M: tl.constexpr, K: tl.constexpr, 
                  N: tl.constexpr, EPS: tl.constexpr, BLOCK_I: tl.constexpr,
                  BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_batch = tl.program_id(2)
    
    i_batch_offset = off_batch * stride_ib
    # y_batch_offset = off_batch * stride_yb
    o_batch_offset = off_batch * stride_ob

    offs_am = (pid_m * BLOCK_I + tl.arange(0, BLOCK_I)) % M
    offs_bn = (pid_n * BLOCK_J + tl.arange(0, BLOCK_J)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    I_block_ptr = I + i_batch_offset + (offs_am[:, None] * stride_im + offs_k[None, :] * stride_ik)
    W1_block_ptr = W1 + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    W3_block_ptr = W3 + (offs_k[:, None] * stride_w3k + offs_bn[None, :] * stride_w3n)
    
    # I_block_ptr = tl.make_block_ptr(
    #     base=I + i_batch_offset,
    #     shape=(M, K),
    #     strides=(stride_im, stride_ik),
    #     offsets=(pid_m * BLOCK_I, 0),
    #     block_shape=(BLOCK_I, BLOCK_K),
    #     order=(1, 0)
    # )
    
    # W1_block_ptr = tl.make_block_ptr(
    #     base=W1, #+ y_batch_offset,
    #     shape=(K, N),
    #     strides=(stride_w1k, stride_w1n),
    #     offsets=(0, pid_n * BLOCK_J),
    #     block_shape=(BLOCK_K, BLOCK_J),
    #     order=(1, 0)
    # )

    # W3_block_ptr = tl.make_block_ptr(
    #     base=W3, #+ y_batch_offset,
    #     shape=(K, N),
    #     strides=(stride_w3k, stride_w3n),
    #     offsets=(0, pid_n * BLOCK_J),
    #     block_shape=(BLOCK_K, BLOCK_J),
    #     order=(1, 0)
    # )
    
    rms_w_ptrs = RMS_W + tl.arange(0, BLOCK_K)[None, :] * stride_rk
    
    acc_w1 = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    acc_w3 = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    temp = tl.zeros((BLOCK_I, BLOCK_K), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        i = tl.load(I_block_ptr)
        
        temp += i * i
        rms_w = tl.load(rms_w_ptrs)
        i = i * rms_w
        
        w1 = tl.load(W1_block_ptr)
        w3 = tl.load(W3_block_ptr)
        
        acc_w1 += tl.dot(i, w1)
        acc_w3 += tl.dot(i, w3)
        
        I_block_ptr += BLOCK_K * stride_ik
        W1_block_ptr += BLOCK_K * stride_w1k
        W3_block_ptr += BLOCK_K * stride_w3k

        rms_w_ptrs += BLOCK_K * stride_rk

    temp_mean = tl.sum(temp, axis=1) / K + EPS
    norm = tl.math.rsqrt(temp_mean)
    acc_w1 = acc_w1 * norm[:, None]
    acc_w3 = acc_w3 * norm[:, None]
    
    # acc_w1 = acc_w1.to(tl.float32)
    acc_w1 = acc_w1 * tl.sigmoid(acc_w1)
    # acc_w1 = acc_w1.to(tl.float16)
    acc = acc_w1 * acc_w3

    offs_outm = pid_m * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_outn = pid_n * BLOCK_J + tl.arange(0, BLOCK_J)
    O_block_ptr = Out + o_batch_offset + (stride_om * offs_outm[:, None] + stride_on * offs_outn[None, :])
    out_mask = (offs_outm[:, None] < M) & (offs_outn[None, :] < N)
    tl.store(O_block_ptr, acc, mask=out_mask)


def kernel_ff(i: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, rms_weight: torch.Tensor):
    assert i.shape[2] == w1.shape[1] == w3.shape[1] == rms_weight.shape[0]
    w1 = w1.T
    w3 = w3.T
    b, m, k = i.shape
    n = w1.shape[-1]
    
    output = torch.empty((b, m, n), dtype=torch.float16, device='cuda')
    BLOCK_I=16
    BLOCK_J=32
    BLOCK_K=32
    assert i.is_cuda and w1.is_cuda and output.is_cuda and rms_weight.is_cuda
    grid = lambda META: (triton.cdiv(m, META['BLOCK_I']), triton.cdiv(n, META['BLOCK_J']), b)
    print(grid({"BLOCK_I":BLOCK_I, "BLOCK_J":BLOCK_J}))
    matmul_kernel[grid](
        i, w1, w3, output, rms_weight,
        i.stride(0), i.stride(1), i.stride(2), 
        w1.stride(0), w1.stride(1),
        w3.stride(0), w3.stride(1),
        output.stride(0), output.stride(1), output.stride(2), 
        rms_weight.stride(0),
        b, m, k, n, EPS=1e-6, BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K
    )
    return output
    
ff = TransformerBlock().cuda()
i = torch.randn(1, 16, 4096).cuda()
st = time.time()
op = ff(i)
print("torch: ", time.time()-st)
# torch.cuda.empty_cache()
st = time.time()
out_triton = kernel_ff(i, ff.feed_forward.w1.weight, ff.feed_forward.w3.weight, ff.ffn_norm.weight)@ff.feed_forward.w2.weight.T
print("triton: ", time.time()-st)
