import gc

import torch


def flush():
    gc.collect()
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)
    torch.cuda.reset_peak_memory_stats(device=device)
    torch.cuda.synchronize(device=device)

flush()