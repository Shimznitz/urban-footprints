# src/model/train.py
import json
import time

import segmentation_models_pytorch as smp
import torch

from src.model.loss import PixelMeter, loss_fn


def build_model(in_channels=8, pretrained=True):
    return smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet" if pretrained else None,  # transfer learning
        in_channels=in_channels,                             # 8 multispectral bands
        classes=1,                                           # building score per pixel
    )


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    meter, total, n = PixelMeter(), 0.0, 0
    for x, y, v in loader:
        x, y, v = x.to(device), y.to(device), v.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        total += loss_fn(logits, y, v).item() * x.size(0)
        n += x.size(0)
        meter.update(logits, y, v, thr)
    return total / n, meter.compute()


def fit(model, train_loader, val_loader, device, ckpt_path, phases):
    """phases: list of dicts: name, epochs, lr, freeze_encoder, optional patience."""
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    best_iou, history = -1.0, []

    for ph in phases:
        for p in model.encoder.parameters():
            p.requires_grad = not ph["freeze_encoder"]
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=ph["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ph["epochs"])
        patience, bad = ph.get("patience", 10**9), 0

        for epoch in range(ph["epochs"]):
            t0 = time.time()
            model.train()
            if ph["freeze_encoder"]:
                model.encoder.eval()  # keep the frozen encoder's BatchNorm statistics fixed
            run, n = 0.0, 0
            for x, y, v in train_loader:
                x, y, v = x.to(device), y.to(device), v.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(x)
                loss = loss_fn(logits.float(), y, v)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run += loss.item() * x.size(0)
                n += x.size(0)
            sched.step()

            val_loss, m = evaluate(model, val_loader, device)
            secs = time.time() - t0
            history.append(dict(phase=ph["name"], epoch=epoch + 1, train_loss=run / n,
                                val_loss=val_loss, secs=secs, **m))
            print(f"[{ph['name']}] epoch {epoch + 1}/{ph['epochs']} | "
                  f"train loss {run / n:.3f} | val loss {val_loss:.3f} | "
                  f"val IoU {m['iou']:.3f} | P {m['precision']:.3f} R {m['recall']:.3f} "
                  f"F1 {m['f1']:.3f} | {secs:.0f}s")

            if m["iou"] > best_iou:
                best_iou, bad = m["iou"], 0
                torch.save(dict(model=model.state_dict(), val_iou=best_iou,
                                phase=ph["name"], epoch=epoch + 1), ckpt_path)
            else:
                bad += 1
            json.dump(history, open(ckpt_path.replace(".pth", "_history.json"), "w"))
            if bad >= patience:
                print(f"early stop: no val IoU gain for {patience} epochs")
                break
    return history