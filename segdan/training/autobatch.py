import os
import torch
import logging

from segdan.exceptions.exceptions import NoValidAutobatchConfigException
import segdan.utils.constants

logger = logging.getLogger(__name__)

def calculate_model_size(model: torch.nn.Module):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_bytes = param_size + buffer_size
    total_size_gb = total_size_bytes / (1024 ** 3)  
    return total_size_gb
    

def profile_memory(img, model, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.train()
    
    gb = 1 << 30
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    img = img.to(device)
    out = model(img)  
    loss = out.mean()
    loss.backward()   
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated(device) / gb
            
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return mem

def autobatch(
    model: torch.nn.Module,
    imgsz=224,
    fraction=0.6
) -> int:
    device = next(model.parameters()).device
        
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    except ValueError as e:
        raise RuntimeError(
            "CUDA operations attempted on CPU. Please run the training on a GPU-enabled environment."
        ) from e
    
    if device.type == 'cuda':
        gb = 1 << 30  # bytes to GiB (1024 ** 3)
        
        d = f"CUDA:{os.getenv('CUDA_VISIBLE_DEVICES', '0').strip()[0]}"  
        properties = torch.cuda.get_device_properties(device)  # device properties
        t = properties.total_memory / gb  # GiB total
        r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
        a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
        f = t - (r + a)  # GiB free
        model_size = calculate_model_size(model)
        usable_mem = f * fraction
        
        if f < 0 or usable_mem < 0:
            raise NoValidAutobatchConfigException(usable_mem)
            
        if model_size > usable_mem:
            raise NoValidAutobatchConfigException(usable_mem, f"Model size ({model_size:.2f} GB) exceeds usable memory ({usable_mem:.2f} GB).")
            
        logger.info(f"{d} device properties (GiB): ")
        logger.info(f"    Total memory: {t:.2f}")
        logger.info(f"    Reserved memory: {r:.2f}")
        logger.info(f"    Allocated memory: {a:.2f}")
        logger.info(f"    Free memory: {f:.2f}")
        logger.info(f"    Model size: {model_size:.2f}")
        logger.info(f"    Usable memory: {usable_mem:.2f}")
    else: 
        usable_mem = None 
        
    batch_sizes = sorted(segdan.utils.constants.AUTOBATCH_SIZES, reverse=True)

    best_batch = None
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        imgs = torch.empty(bs, 3, imgsz, imgsz, device=device)
        
        try:
            mem_used = profile_memory(imgs, model, device)
            logger.info(f"Testing batch_size={bs}, mem={mem_used:.2f}GB")
        except Exception as e:
            logger.error(f"  Error profiling batch_size={bs}: {e}")
            continue
            
        if usable_mem and mem_used <= usable_mem:
            best_batch = bs
            break   
        
    if best_batch is None:
        raise NoValidAutobatchConfigException(usable_mem)

    logger.info(f"Best batch size found:{best_batch}")
    return best_batch 