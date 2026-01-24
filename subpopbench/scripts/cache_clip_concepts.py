#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

import open_clip

from subpopbench.dataset import datasets
from subpopbench.dataset.fast_dataloader import FastDataLoader
from subpopbench import hparams_registry


def load_concept_meta(meta_path: str, include_all: bool = True) -> Tuple[List[str], torch.Tensor]:
    """
    Returns:
      concept_text: list[str] length K_all (if include_all=True) else K_used
      use_mask: BoolTensor length K_all (always returned)
    """
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    concept_text = meta["concept_text"]
    use_in_training = meta.get("use_in_training", [True] * len(concept_text))
    use_mask = torch.as_tensor(use_in_training, dtype=torch.bool)

    assert len(concept_text) == int(use_mask.numel()), "meta length mismatch"

    if include_all:
        return concept_text, use_mask
    else:
        filtered = [t for t, u in zip(concept_text, use_mask.tolist()) if u]
        return filtered, use_mask


@torch.no_grad()
def encode_text_features(model, tokenizer, texts: List[str], device: str, batch_size: int = 64) -> torch.Tensor:
    feats = []
    for s in range(0, len(texts), batch_size):
        chunk = texts[s:s + batch_size]
        toks = tokenizer(chunk).to(device)
        z = model.encode_text(toks)
        z = F.normalize(z, dim=-1)
        feats.append(z)
    return torch.cat(feats, dim=0)  # [K, D]


@torch.no_grad()
def encode_image_features(model, x: torch.Tensor, device: str) -> torch.Tensor:
    z = model.encode_image(x.to(device))
    z = F.normalize(z, dim=-1)
    return z  # [B, D]


def main():
    parser = argparse.ArgumentParser(description="Cache OpenCLIP concept probabilities for a SubpopBench dataset split.")
    parser.add_argument("--dataset", type=str, default="Waterbirds", choices=datasets.DATASETS)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--split", type=str, required=True, choices=["tr", "va", "te"])
    parser.add_argument("--meta", type=str, required=True, help="concept meta JSON (concept_text order defines columns)")
    parser.add_argument("--out", type=str, required=True, help="output .pt path")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")

    # OpenCLIP settings
    parser.add_argument("--clip-model", type=str, default="RN50")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--text-batch-size", type=int, default=64)

    # Output settings
    parser.add_argument("--save-only-used", action="store_true",
                        help="If set, only save concepts where use_in_training=True (drops excluded dims).")
    parser.add_argument("--fp16", action="store_true", help="Use float16 inference (faster on GPU).")

    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # Load concept list (include all by default so excluded dims still exist in cache)
    concept_text, use_mask_full = load_concept_meta(args.meta, include_all=not args.save_only_used)
    K = len(concept_text)

    # Use dataset default hparams (keeps transforms consistent with SubpopBench defaults)
    hparams = hparams_registry.default_hparams("ERM", args.dataset)

    # Build dataset + loader
    dset_cls = vars(datasets)[args.dataset]
    dset = dset_cls(args.data_dir, args.split, hparams)
    loader = FastDataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # ------------------------------------------------------------------
    # CRITICAL FIX:
    # The dataset-provided index `i` may NOT be 0..len(dset)-1.
    # It can be a global/original-id index space (e.g. max ~5792 for va).
    # So size the cache by max(i)+1, and write to rows at those indices.
    # ------------------------------------------------------------------
    max_i = -1
    with torch.no_grad():
        for i, x, y, a in loader:
            mi = int(i.max().item())
            if mi > max_i:
                max_i = mi
    assert max_i >= 0, "Empty loader? (No samples yielded.)"

    scores_dtype = torch.float16 if args.fp16 else torch.float32
    scores = torch.empty((max_i + 1, K), dtype=scores_dtype)
    scores.zero_()  # deterministic for any unused rows

    # Load OpenCLIP
    model, _, _ = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrained)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model = model.to(device)
    model.eval()

    if args.fp16 and device != "cpu":
        model = model.half()

    # Precompute text features
    text_feats = encode_text_features(model, tokenizer, concept_text, device=device, batch_size=args.text_batch_size)
    if args.fp16 and device != "cpu":
        text_feats = text_feats.half()

    # CLIP scaling
    logit_scale = getattr(model, "logit_scale", None)
    scale = float(logit_scale.exp().item()) if logit_scale is not None else 1.0

    # Score all images (second pass)
    with torch.no_grad():
        for i, x, y, a in loader:
            if args.fp16 and device != "cpu":
                x = x.half()
            img_feats = encode_image_features(model, x, device=device)    # [B, D]
            logits = (scale * img_feats @ text_feats.T)                  # [B, K]
            probs = torch.sigmoid(logits).detach().cpu()                 # [B, K]

            idx = i.long().cpu()
            if int(idx.max()) >= scores.shape[0]:
                raise RuntimeError(
                    f"Internal error: idx.max()={int(idx.max())} >= scores rows={scores.shape[0]}"
                )

            scores.index_copy_(0, idx, probs.to(scores.dtype))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    save_obj: Dict[str, Any] = {
        "concepts": scores,  # [max_i+1, K]
        "meta_path": args.meta,
        "dataset": args.dataset,
        "split": args.split,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "K": K,
        "save_only_used": bool(args.save_only_used),
        # extra info so downstream code can sanity-check expectations
        "index_space": "dataset_i",
        "max_i": int(max_i),
        "len_dataset": int(len(dset)),
        "dtype": str(scores.dtype),
    }
    torch.save(save_obj, args.out)
    print(f"Saved concept cache -> {args.out}")
    print(f"  shape: {tuple(scores.shape)}  dtype: {scores.dtype}  max_i={max_i}  len(dset)={len(dset)}  K={K}")


if __name__ == "__main__":
    main()
