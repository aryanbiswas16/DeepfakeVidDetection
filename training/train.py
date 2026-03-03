import os
import sys
# Add project root to python path so we can import from datasets, models, utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from datasets.paired_dataset import PairedDataset
from datasets.faceforensics import FaceForensicsDataset
from datasets.celeb_df import CelebDFDataset
from models.detector import Detector
from utils.preprocess import build_preprocess, build_training_preprocess


def metric_loss(embeddings, labels, margin=0.6):
    """Metric loss with improved margin."""
    cos_sim = torch.nn.functional.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
    )
    targets = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
    loss = (targets * torch.nn.functional.relu(cos_sim + margin)).mean()
    return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Use robust augmentation for training
    train_preprocess = build_training_preprocess(image_size=224)
    # Use standard preprocess for validation
    val_preprocess = build_preprocess(image_size=224)


    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", str(max(2, batch_size * 2))))

    # On Windows, DataLoader multiprocessing can spawn with an unexpected interpreter
    # if the environment isn't activated consistently. Defaulting to 0 keeps it robust.
    default_num_workers = "0" if os.name == "nt" else "4"
    num_workers = int(os.environ.get("NUM_WORKERS", default_num_workers))


    
    train_datasets = []
    val_datasets = []

    # 1. Load FF++ for Training
    ff_root = "data/processed_ff"
    if os.path.exists(ff_root):
        print(f"Loading FaceForensics++ from {ff_root}...")
        # Load all available methods
        manipulated_path = os.path.join(ff_root, "manipulated_sequences")
        if os.path.exists(manipulated_path):
            for method in os.listdir(manipulated_path):
                print(f"Adding FF++ method: {method}")
                ds = FaceForensicsDataset(ff_root, method=method, split="all", transform=train_preprocess)
                if len(ds) > 0:
                    train_datasets.append(ds)
        else:
            # Fallback if structure isn't fully there yet or just Deepfakes
            ds = FaceForensicsDataset(ff_root, method="Deepfakes", split="all", transform=train_preprocess)
            if len(ds) > 0: train_datasets.append(ds)
    else:
        print(f"Warning: FF++ root not found at {ff_root}")

    # 2. Load Celeb-DF for Validation
    celeb_root = "data/processed_celeb"
    if os.path.exists(celeb_root):
        print(f"Loading Celeb-DF from {celeb_root}...")
        # Use ONLY Celeb-DF test split for true generalization evaluation
        # If 'test' folder doesn't exist/is empty, fallback to 'val'
        ds_c_test = CelebDFDataset(celeb_root, split="test", transform=val_preprocess)
        if len(ds_c_test) > 0:
            val_datasets.append(ds_c_test)
            print("Using Celeb-DF 'test' split for validation.")
        else:
            ds_c_val = CelebDFDataset(celeb_root, split="val", transform=val_preprocess)
            if len(ds_c_val) > 0:
                val_datasets.append(ds_c_val)
                print("Using Celeb-DF 'val' split for validation (test split empty/missing).")
    else:
        print(f"Warning: Celeb-DF root not found at {celeb_root}")

    if not train_datasets:
        print("No training data found. Please run scripts/process_data.py first.")
        return

    dataset = ConcatDataset(train_datasets)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Training on {len(dataset)} frames (FF++)")

    val_loader = None
    if val_datasets:
        val_dataset = ConcatDataset(val_datasets)
        val_loader = DataLoader(
            val_dataset,
    # Use FocalLoss to focus on hard examples (like NeuralTextures)
    criterion = FocalLoss(gamma=2.0e,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"Validation enabled: {len(val_dataset)} frames (Celeb-DF)")
    else:
        print("Validation disabled (no validation data found)")

    detector = Detector(device=device)
    
    # Unfreeze the last block of the encoder for better fine-tuning
    
    if hasattr(detector.encoder.model, 'blocks'):
        last_block = detector.encoder.model.blocks[-1]
        for p in last_block.parameters():
            p.requires_grad = True
    
    # Collect all trainable parameters
    params = [p for p in detector.parameters() if p.requires_grad]

    optimizer = optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20) 
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    print("Starting paired training loop...")
    detector.head.train()
    detector.encoder.train()

    def _evaluate(det: Detector, loader: DataLoader):
        det.eval()
        all_labels = []
        all_probs = []
        correct = 0
        total = 0
        
        # Add progress bar for validation
        val_bar = tqdm(loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for real_imgs, fake_imgs, _ in val_bar:
                imgs = torch.cat([real_imgs, fake_imgs], dim=0).to(device)
                labels = torch.cat(
                    [torch.zeros(real_imgs.size(0)), torch.ones(fake_imgs.size(0))], dim=0
                ).long().to(device)

                logits, _ = det.forward(imgs)
                probs = torch.softmax(logits, dim=-1)[:, 1]

                preds = (probs > 0.5).long()
                correct += int((preds == labels).sum().item())
                total += int(labels.numel())

                all_labels.extend(labels.detach().cpu().tolist())
                all_probs.extend(probs.detach().cpu().tolist())

        acc = float(correct / max(total, 1))

        auroc = None
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore

            auroc = float(roc_auc_score(all_labels, all_probs))
        except Exception as e:
            print(f"AUROC unavailable: {e}")

        return acc, auroc


    best_val_auroc = float("-inf")
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for real_imgs, fake_imgs, _ in progress_bar:
            imgs = torch.cat([real_imgs, fake_imgs], dim=0).to(device)
            labels = torch.cat(
                [torch.zeros(real_imgs.size(0)), torch.ones(fake_imgs.size(0))],
                dim=0
            ).long().to(device)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits, embeddings = detector.forward(imgs)
                ce = criterion(logits, labels)
                ml = metric_loss(embeddings, labels, margin=0.6) 
                loss = ce + 0.5 * ml   

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loader is not None:
            val_acc, val_auroc = _evaluate(detector, val_loader)
            if val_auroc is None:
                print(f"Val Acc: {val_acc:.4f}")
            else:
                print(f"Val Acc: {val_acc:.4f} | Val AUROC: {val_auroc:.4f}")

        # Save best checkpoint by val AUROC when available, otherwise save last epoch
        should_save = (val_loader is None)
        if val_loader is not None:
            # only save by AUROC when AUROC exists
            if val_auroc is not None and val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                should_save = True

        if should_save:
            os.makedirs("training/weights", exist_ok=True)
            save_path = "training/weights/dinov3_best_v4.pth"
            
            state = {
                "head": detector.head.state_dict(),
            }
            
            # Save ALL trainable params from encoder (LayerNorms + Unfrozen Blocks)
            encoder_state = {}
            try:
                for k, v in detector.encoder.model.named_parameters():
                    if v.requires_grad:
                        encoder_state[k] = v.detach().cpu()
            except Exception as e:
                print(f"Warning: Could not extract encoder state: {e}")
            
            if encoder_state:
                state["encoder"] = encoder_state
            
            torch.save(state, save_path)
            
            if val_loader is None:
                print(f"✓ Saved checkpoint to {save_path}")
            else:
                print(f"✓ Saved best checkpoint (Val AUROC={best_val_auroc:.4f})")


if __name__ == '__main__':
    train()
