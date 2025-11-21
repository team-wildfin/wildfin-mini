import time
from contextlib import contextmanager
import io   
import torch

@contextmanager
def step_timer(name, verbose=True):
    if verbose: print(f"[{name}] started")
    start = time.time()
    yield
    end = time.time()
    if verbose: print(f"[{name}] took {end - start:.6f} seconds")

def print_info(tensor):
    print("type:", type(tensor))
    print("dtype:", tensor.dtype)
    print("shape:", tensor.shape)
    print("device:", tensor.device)
    print("requires_grad:", tensor.requires_grad)
    print("is_contiguous:", tensor.is_contiguous())
    print("stride:", tensor.stride())
    print("storage_offset:", tensor.storage_offset())

def serialized_size(t):
    buf = io.BytesIO()
    torch.save(t, buf)
    return len(buf.getvalue())