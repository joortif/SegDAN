# install_pytorch.py
import subprocess
import sys
import shutil

CUDA_VERSION = "cu118"
TORCH_VERSION = "2.1.0"
TORCHVISION_VERSION = "0.16.0"
TORCHAUDIO_VERSION = "2.7.2"

def has_nvidia_gpu():
    return shutil.which("nvidia-smi") is not None

def install(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def install_gpu():
    print("Installing PyTorch with GPU (CUDA 11.8)...")
    install([
        "pip", "install",
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"torchaudio=={TORCHAUDIO_VERSION}",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])

def install_cpu():
    print("Installing PyTorch CPU version...")
    install([
        "pip", "install",
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"torchaudio=={TORCHAUDIO_VERSION}"
    ])

def main():
    try:
        import torch
        import torchvision
        print("PyTorch and torchvision already installed.")
    except ImportError:
        print("PyTorch not found. Installing...")

        if has_nvidia_gpu():
            try:
                install_gpu()
                return
            except Exception as e:
                print(f"GPU install failed: {e}")
        
        install_cpu()

if __name__ == "__main__":
    main()
