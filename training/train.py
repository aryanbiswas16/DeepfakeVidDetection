import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.paired_dataset import PairedDataset
from models.detector import Detector
from utils.preprocess import build_preprocess


def metric_loss(embeddings, labels, margin=0.3):
    # embeddings: [B, D]
    # labels: [B]
    # compute cosine similarity matrix
    cos_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    targets = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
    loss = (targets * torch.nn.functional.relu(cos_sim + margin)).mean()
    return loss


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    preprocess = build_preprocess(image_size=224)

    # Update these paths as needed
    real_dir = "data/train/real"
    fake_dir = "data/train/fake"

    # Optional validation split (recommended)
    val_real_dir = "data/val/real"
    val_fake_dir = "data/val/fake"

    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print("No paired training data found. Create 'data/train/real' and 'data/train/fake' with matching filenames.")
        return

    dataset = PairedDataset(real_dir, fake_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    has_val = os.path.exists(val_real_dir) and os.path.exists(val_fake_dir)
    val_loader = None
    if has_val:
        try:
            val_dataset = PairedDataset(val_real_dir, val_fake_dir, transform=preprocess)
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
            print(f"Validation enabled: {len(val_dataset)} paired frames")
        except Exception as e:
            print(f"Validation disabled (failed to build val dataset): {e}")
            val_loader = None

    detector = Detector(device=device)
    # Only train head + any LayerNorm params that were left unfrozen in encoder
    params = [p for p in detector.head.parameters() if p.requires_grad]
    # Also include encoder params that require grad (LayerNorm tuning)
    params += [p for p in detector.encoder.model.parameters() if p.requires_grad]

    optimizer = optim.Adam(params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    print("Starting paired training loop...")
    detector.head.train()
    detector.encoder.train()

    def _evaluate(det: Detector, loader: DataLoader):
        det.eval()
        all_labels = []
        all_probs = []
        correct = 0
        total = 0
        with torch.no_grad():
            for real_imgs, fake_imgs, _ in loader:
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
            print(f"AUROC unavailable (install scikit-learn to enable): {e}")

        return acc, auroc


    best_val_auroc = float("-inf")
    for epoch in range(5):
        total_loss = 0.0
        for real_imgs, fake_imgs, _ in dataloader:
            # real_imgs, fake_imgs: (B, C, H, W) tensors already preprocessed
            imgs = torch.cat([real_imgs, fake_imgs], dim=0).to(device)
            labels = torch.cat([torch.zeros(real_imgs.size(0)), torch.ones(fake_imgs.size(0))], dim=0).long().to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits, embeddings = detector.forward(imgs)
                ce = criterion(logits, labels)
                ml = metric_loss(embeddings, labels)
                loss = ce + 0.5 * ml

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

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
            save_path = "training/weights/dinov2_best.pth"
            state = {
                "head": detector.head.state_dict(),
            }
            try:
                state["encoder"] = {
                    k: v
                    for k, v in detector.encoder.model.state_dict().items()
                    if "norm" in k.lower() or "ln" in k.lower()
                }
            except Exception:
                pass

            torch.save(state, save_path)
            if val_loader is None:
                print(f"Saved weights to {save_path}")
            else:
                print(f"Saved best weights to {save_path} (val_auroc={best_val_auroc:.4f})")


if __name__ == '__main__':
    train()
