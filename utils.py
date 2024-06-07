import gc
import torch


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    

def get_device(device):
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device