import sys
import torch
import torch._dynamo

def verify_torch():
    print(f"Python version: {sys.version}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    print("\nChecking Torch Dynamo internals...")
    try:
        from torch._C._dynamo.eval_frame import skip_code
        print("✓ Successfully imported skip_code from torch._C._dynamo.eval_frame")
    except ImportError as e:
        print(f"✗ ImportError: {e}")
        print("\nPossible fix: Clean reinstall of torch, torchvision, and torchaudio.")
        return False

    if hasattr(torch, "_dynamo"):
        print("✓ torch._dynamo is available")
    else:
        print("✗ torch._dynamo is not available in this torch version")
        return False

    print("\nEnvironment check passed! ✓")
    return True

if __name__ == "__main__":
    if not verify_torch():
        sys.exit(1)
