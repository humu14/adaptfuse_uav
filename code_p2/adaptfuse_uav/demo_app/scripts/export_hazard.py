"""
Build the zero-shot hazard weights used by engine/hazard.py:

    weights/hazard_clip/      ViT-B/16 vision tower + projection, fp16, shards < 90 MB
    weights/hazard_text.pt    L2-normalised text embedding per HAZARD_EVENTS entry
                              (mean over prompts x templates) + logit scale

Source: openai/clip-vit-base-patch16 from the Hugging Face hub (downloaded once).
Re-run after editing HAZARD_EVENTS / HAZARD_TEMPLATES in engine/labels.py, then re-run
scripts/eval_hazard.py so the calibration matches.

    python scripts/export_hazard.py            # from demo_app/
"""

from __future__ import annotations

import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
import engine  # noqa: E402,F401
import torch  # noqa: E402

from engine.labels import HAZARD_EVENTS, HAZARD_NAMES, HAZARD_TEMPLATES  # noqa: E402
from engine.models import WEIGHTS_DIR  # noqa: E402

REPO_ID = "openai/clip-vit-base-patch16"


def load_clip():
    """CLIPModel from the hub. The repo ships only pytorch_model.bin; transformers >= 4.50 refuses
    torch.load on torch < 2.6, so load the state dict ourselves with weights_only=True."""
    from huggingface_hub import hf_hub_download
    from transformers import CLIPConfig, CLIPModel, CLIPTokenizer
    cfg = CLIPConfig.from_pretrained(REPO_ID)
    model = CLIPModel(cfg)
    sd = torch.load(hf_hub_download(REPO_ID, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "position_ids" not in k]
    if missing:
        raise RuntimeError(f"missing CLIP weights: {missing[:5]}")
    return model.eval(), CLIPTokenizer.from_pretrained(REPO_ID)


@torch.no_grad()
def text_embeddings(model, tok) -> torch.Tensor:
    feats = []
    for ev in HAZARD_EVENTS:
        prompts = [t.format(p) for p in ev["prompts"] for t in HAZARD_TEMPLATES]
        enc = tok(prompts, padding=True, return_tensors="pt")
        e = model.get_text_features(**enc).float()
        e = e / e.norm(dim=-1, keepdim=True)
        m = e.mean(0)
        feats.append(m / m.norm())
    return torch.stack(feats)


def main():
    from transformers import CLIPVisionModelWithProjection
    model, tok = load_clip()
    text = text_embeddings(model, tok)
    torch.save({"names": HAZARD_NAMES, "text": text.half(), "logit_scale": float(model.logit_scale.exp()),
                "source": REPO_ID, "templates": HAZARD_TEMPLATES,
                "prompts": {e["name"]: e["prompts"] for e in HAZARD_EVENTS}},
               WEIGHTS_DIR / "hazard_text.pt")

    vis = CLIPVisionModelWithProjection(model.config.vision_config)
    vis.vision_model.load_state_dict(model.vision_model.state_dict())
    vis.visual_projection.load_state_dict(model.visual_projection.state_dict())
    out = WEIGHTS_DIR / "hazard_clip"
    vis.half().save_pretrained(out, max_shard_size="90MB", safe_serialization=True)
    size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"[ok] {out} ({size:.0f} MB), hazard_text.pt ({len(HAZARD_NAMES)} events, scale {model.logit_scale.exp():.1f})")


if __name__ == "__main__":
    main()
