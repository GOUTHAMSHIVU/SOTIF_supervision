import torch
import h5py
from pathlib import Path
from PIL import Image
from torchvision import transforms
from src.models.backbone import ResNet50FeatureExtractor

def run_sanity_check():
    print("=" * 60)
    print("RUNNING FEATURE EXTRACTION SANITY CHECK")
    print("=" * 60)

    # 1. Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[CHECK 1/5] Compute device: {device} ({gpu_name})")

    # 2. Dataset path & sample loading
    data_dir = Path("data/raw/a2d2")
    if not data_dir.exists():
        data_dir = Path("D:/SOTIF_supervision_workspace/data/raw/a2d2")
    
    sample_paths = sorted(list(data_dir.rglob("*.png")))[:16]
    assert len(sample_paths) == 16, f"Expected 16 images, found {len(sample_paths)}"
    print(f"[CHECK 2/5] Successfully found sample images.")
    print(f"            Sample 0: {sample_paths[0].name}")

    # 3. Preprocessing transformation
    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch_tensor = torch.stack([transform(Image.open(p).convert("RGB")) for p in sample_paths]).to(device)
    print(f"[CHECK 3/5] Input batch tensor shape: {tuple(batch_tensor.shape)}")

    # 4. ResNet-50 feature extraction pass
    model = ResNet50FeatureExtractor(pretrained=True).to(device)
    model.eval()
    with torch.no_grad():
        feats = model(batch_tensor)
    
    print(f"[CHECK 4/5] Output feature tensor shape: {tuple(feats.shape)}")
    assert feats.shape == (16, 2048), f"Shape mismatch: expected (16, 2048), got {feats.shape}"

    # 5. HDF5 read/write integrity check
    test_h5 = Path("data/metadata/test_sanity.h5")
    test_h5.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(test_h5, "w") as f:
        f.create_dataset("features", data=feats.cpu().numpy(), dtype="float32")

    with h5py.File(test_h5, "r") as f:
        stored_shape = tuple(f["features"].shape)
        assert stored_shape == (16, 2048), f"HDF5 mismatch: expected (16, 2048), got {stored_shape}"

    test_h5.unlink()
    print(f"[CHECK 5/5] HDF5 write/read roundtrip passed.")

    print("=" * 60)
    print("[PASSED] ALL SANITY CHECKS PASSED. Ready for full extraction run.")
    print("=" * 60)

if __name__ == "__main__":
    run_sanity_check()