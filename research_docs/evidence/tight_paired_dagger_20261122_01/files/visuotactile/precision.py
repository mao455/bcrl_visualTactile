"""Explicit arithmetic policy shared by the new StateDP audit/train/deploy run."""
import torch


def state_fp32():
    # Match the existing trainer, including CuDNN's GRU/convolution path.
    # No history, model parameter, sampler or controller setting changes here.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return snapshot()


def snapshot():
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                autocast_cuda=torch.is_autocast_enabled(), torch_version=torch.__version__)
