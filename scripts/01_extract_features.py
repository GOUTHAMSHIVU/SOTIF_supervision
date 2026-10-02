# src/data/extract_features.py
from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.models.backbone import ResNet50FeatureExtractor

class A2D2ImageFolder(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.image_paths = sorted(list(Path(root_dir).rglob("*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(img_path.name)

def run_extraction(data_dir: str, output_h5: str, batch_size: int = 64, num_workers: int = 4, limit: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")

    # Standard ImageNet normalization and resolution scaling
    transform = transforms.Compose([
        transforms.Resize((256, 512)),   # Preserves wide automotive aspect ratio
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = A2D2ImageFolder(root_dir=data_dir, transform=transform)
    total_samples = len(dataset)
    print(f"[INFO] Found {total_samples} frames in {data_dir}")

    if total_samples == 0:
        raise RuntimeError("No .png files found. Verify the data download path.")

    # Slice dataset if a limit is provided for sanity checks
    if limit is not None and limit > 0:
        actual_limit = min(limit, total_samples)
        dataset = Subset(dataset, range(actual_limit))
        print(f"[INFO] Dry run enabled: processing only first {actual_limit} frames.")

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    extractor = ResNet50FeatureExtractor().to(device)

    out_path = Path(output_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-allocate contiguous chunked HDF5 dataset
    with h5py.File(output_h5, "w") as h5_file:
        feat_ds = h5_file.create_dataset(
            "features",
            shape=(total_samples, 2048),
            dtype="float32",
            chunks=(min(1024, total_samples), 2048)
        )
        # Store file identifiers to maintain label traceability
        filename_ds = h5_file.create_dataset(
            "filenames",
            shape=(total_samples,),
            dtype=h5py.string_dtype(encoding="utf-8")
        )

        cursor = 0
        for images, names in tqdm(loader, desc="Extracting 2048-d features"):
            images = images.to(device, non_blocking=True)
            embeddings = extractor(images).cpu().numpy()

            batch_len = embeddings.shape[0]
            feat_ds[cursor : cursor + batch_len] = embeddings
            filename_ds[cursor : cursor + batch_len] = names
            cursor += batch_len

    print(f"[SUCCESS] Features saved to: {output_h5} ({Path(output_h5).stat().st_size / (1024**2):.2f} MB)")

        # At the end of run_extraction(...) in src/data/extract_features.py:
    from utils.manifest import write_artifact_manifest

    # ... existing HDF5 writing finishes here ...

    manifest_path = "data/metadata/nominal_train_manifest_post.json"
    write_artifact_manifest(
        h5_path=output_h5,
        raw_data_dir=data_dir,
        output_manifest_path=manifest_path
    )