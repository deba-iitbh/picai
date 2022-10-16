import numpy as np
import os
import torch


def compute_spec_for_run(args, train_device='cuda'):
    """Register GPU/CPU specifications"""
    device = torch.device(train_device)
    
    # os.sched_getaffinity(0) is not supported by all operating systems
    try:
        args.num_threads = np.min([len(os.sched_getaffinity(0)), args.max_threads])
    except:
        args.num_threads = args.max_threads
    
    print("="*100)
    print("GPUs:", torch.cuda.get_device_name(0))
    print("CPU Threads:", args.num_threads)
    return device, args
