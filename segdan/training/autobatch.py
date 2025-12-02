import os
from typing import Any, Callable, Optional
import torch
import logging

from transformers import OneFormerProcessor

from segdan.exceptions.exceptions import NoValidAutobatchConfigException
import segdan.utils.constants
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)

def _extract_first_tensor(obj):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            t = _extract_first_tensor(v)
            if t is not None:
                return t
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _extract_first_tensor(v)
            if t is not None:
                return t
    for attr in ["logits", "out", "preds", "class_queries_logits", "masks_queries_logits", "last_hidden_state"]:
        if hasattr(obj, attr):
            v = getattr(obj, attr)
            if torch.is_tensor(v) or (isinstance(v, (list, tuple, dict)) and _extract_first_tensor(v) is not None):
                return _extract_first_tensor(v)
    return None

def _tensorlist_to_hwc_numpy_list(tensor: torch.Tensor):
    # tensor: (B, C, H, W) o (C, H, W)
    if tensor.dim() == 4:
        # Batch
        return [img.permute(1, 2, 0).cpu().numpy() for img in tensor]
    elif tensor.dim() == 3:
        # Single image
        return [tensor.permute(1, 2, 0).cpu().numpy()]
    else:
        raise ValueError("3D or 4D tensor expected")

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
    

def profile_memory(
    img: Any,
    model: torch.nn.Module,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = smp.losses.DiceLoss,
    device: Optional[torch.device] = None,
    processor = None,
    verbose: bool = True
) -> float:
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gb = 1 << 30

    model.to(device)
    model.train()
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.zero_grad(set_to_none=True)
    
    try:
        inputs_for_processor = img

        if processor is not None:
            args = {
                "images": inputs_for_processor,
                "do_resize": False,
                "do_normalize": True,
                "do_rescale": False,
                "return_tensors": "pt",
            }

            # OneFormer specifics
            if isinstance(processor, OneFormerProcessor):
                if torch.is_tensor(img):
                    bs = img.shape[0]
                else:
                    bs = len(img)

                args["task_inputs"] = ["semantic"] * bs
                processor.image_processor.num_text =1

            if torch.is_tensor(inputs_for_processor):
                images_list = _tensorlist_to_hwc_numpy_list(inputs_for_processor)
                proc_out = processor(images=images_list, do_resize=False, do_normalize=True, return_tensors="pt")
            else:
                proc_out = processor(**args)
            
            img_device = {}
            if proc_out is not None:
                for k, v in proc_out.items():
                    if torch.is_tensor(v):
                        img_device[k] = v.to(device)
                    elif isinstance(v, (list, tuple)):
                        new_list = []
                        for elem in v:
                            if torch.is_tensor(elem):
                                new_list.append(elem.to(device))
                            else:
                                new_list.append(elem)
                        img_device[k] = new_list
                    else:
                        img_device[k] = v
            
        outputs = model(**img_device)

    except Exception as e:
        if verbose:
            err_name = type(e).__name__
            if "out of memory" in str(e).lower():
                try:
                    batch = img.shape[0]
                except Exception:
                    batch = "?"
                logger.info(f"CUDA OOM in forward (batch {batch})")
            else:
                logger.error(f"Forward call failed ({err_name}): {e}")
        raise e
    
    loss = None
    # HF
    if hasattr(outputs, "loss") and getattr(outputs, "loss") is not None:
        loss = outputs.loss
        if verbose:
            logger.info("Using outputs.loss from model.")
    else:
        # SMP
        first_tensor = _extract_first_tensor(outputs)
        if first_tensor is not None and torch.is_tensor(first_tensor):
            loss = first_tensor.mean()
            if verbose:
                logger.info("Using mean(first_tensor) as fallback loss.")
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            if verbose:
                logger.info("No tensor found in outputs and no labels/loss_fn provided: using dummy scalar loss.")

    try:
        loss.backward()
    except Exception as e:
        try:
            loss = loss.float()
            loss.backward()
        except Exception:
            if verbose:
                logger.error("Backward failed:", e)
            raise

    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / gb
        
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    return float(peak)

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

    logger.info(f"Best batch size found: {best_batch}")
    return best_batch 