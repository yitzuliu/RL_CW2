import platform
import os
import torch
import psutil
import numpy as np

_device = None

def get_device():
    global _device
    if _device is not None:
        return _device
    _device = torch.device("cpu")
    device_name = "CPU"
    if torch.cuda.is_available():
        _device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
        torch.backends.cudnn.benchmark = True
    elif (hasattr(torch.backends, "mps") and 
          torch.backends.mps.is_available() and 
          torch.backends.mps.is_built()):
        _device = torch.device("mps")
        device_name = "Apple Silicon (MPS)"
    print(f"Using device: {device_name}")
    return _device

def get_gpu_memory_gb():
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            return None
    return None

def get_system_info():
    info = {
        'os': platform.system(),
        'os_version': platform.version(),
        'cpu_type': platform.processor() or "Unknown",
        'cpu_count': os.cpu_count(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    }
    try:
        info['system_memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)
    except:
        info['system_memory_gb'] = "Unknown"
    if info['cuda_available']:
        try:
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = round(get_gpu_memory_gb(), 2)
        except:
            info['gpu_name'] = "Unknown CUDA device"
            info['gpu_memory_gb'] = "Unknown"
    return info