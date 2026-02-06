#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


from subpopbench.learning.algorithms import CR
from subpopbench.dataset import datasets
from subpopbench.dataset.fast_dataloader import FastDataLoader
from subpopbench import hparams_registry


def trunc(s: str, n: int = 18) -> str:
    s = str(s)
    return s if len(s) <= n else "distinct marks"

def topk_abs(vals: np.ndarray, k: int):
    vals = np.asarray(vals)
    k = min(k, vals.size)
    if k <= 0:
        return np.array([], dtype=int)
    return np.argsort(-np.abs(vals))[:k]

def load_cr_from_checkpoint(model_pkl: str):
    ckpt = torch.load(model_pkl, map_location="cpu", weights_only=False)
    model = CR(
        data_type="images",
        input_shape=tuple(ckpt["model_input_shape"]),
        num_classes=int(ckpt["num_labels"]),
        num_attributes=int(ckpt["num_attributes"]),
        num_examples=0,
        hparams=ckpt["model_hparams"],
    )
    model.load_state_dict(ckpt["model_dict"], strict=True)
    model.eval()
    return model

def get_linear_layer(model):
    lin = getattr(model.classifier, "linear", None)
    if lin is None:
        if isinstance(model.classifier, nn.Linear):
            lin = model.classifier
        else:
            lins = [m for m in model.classifier.modules() if isinstance(m, nn.Linear)]
            assert len(lins) > 0, "No Linear layer found in classifier"
            lin = lins[-1]
    return lin

def load_meta(meta_path: str, model):
    meta = json.load(open(meta_path, "r"))
    concept_text = meta["concept_text"]
    channels = meta.get("channel", ["unknown"] * len(concept_text))

    # EXACT mask used in training (includes any blocking decisions)
    use_mask_full = model._concept_use_mask.detach().cpu().numpy().astype(bool)

    concept_used = [t for t, u in zip(concept_text, use_mask_full) if u]
    channel_used = [ch for ch, u in zip(channels, use_mask_full) if u]

    K_used = int(model.concept_dim_used)
    assert len(concept_used) == K_used, f"mask produced {len(concept_used)} used concepts, expected {K_used}"
    assert len(channel_used) == K_used
    return concept_used, channel_used, use_mask_full

def find_example_by_i(dataset, i_target: int):
    loader = FastDataLoader(dataset=dataset, batch_size=256, num_workers=2)
    for batch in loader:
        i, x, y, a = batch
        i = i.long()
        mask = (i == i_target)
        if mask.any():
            j = int(mask.nonzero(as_tuple=False)[0].item())
            return x[j], int(y[j].item()), int(a[j].item())
    raise ValueError(f"Could not find i={i_target} in this split")

def nice_title(ch: str) -> str:
    return ch.replace("_", " ")


def plot_channel(ax, labels, vals, title, normalize=True):
    vals = np.asarray(vals, dtype=np.float64)

    scale = 1.0
    if normalize:
        denom = float(np.max(np.abs(vals))) if vals.size else 1.0
        denom = max(denom, 1e-12)
        scale = denom
        vals = vals / denom  # now in [-1,1]

    ax.barh(range(len(vals)), vals, height=0.8)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_anchor('C')

    ax.set_title(title.replace("_"," "), fontsize=14, pad=6)

    # keep the x-axis clean
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(axis="x", labelsize=9)

    # symmetric limits
    m = float(np.max(np.abs(vals))) if vals.size else 1.0
    m = max(m, 1e-6)   # prevent collapse
    ax.set_xlim(-1.1*m, 1.1*m)


    # optional small subtitle showing the raw scale
    if normalize:
        ax.text(
            0.98, -0.12,
            f"",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pkl", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="te", choices=["tr", "va", "te"])
    ap.add_argument("--i", type=int, required=True, help="dataset index i (cache index-space)")
    ap.add_argument("--concept_cache", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", default="figs/channel_two.png")
    ap.add_argument("--class_idx", type=int, default=1)
    ap.add_argument("--topk", type=int, default=6)

    # fixed choice per your ask
    ap.add_argument("--ch1", default="target_attributes")
    ap.add_argument("--ch2", default="cooccurring_objects")
    args = ap.parse_args()

    model = load_cr_from_checkpoint(args.model_pkl)

    feature_mode = model.hparams.get("cr_feature_mode", "")
    assert feature_mode == "concat_plus_resid", f"Expected concat_plus_resid, got {feature_mode}"
    assert model.use_concepts, "Model was trained without concepts (use_concepts=False)."

    lin = get_linear_layer(model)
    d = int(model.featurizer.n_outputs)
    K_used = int(model.concept_dim_used)

    # concept cache (logits)
    obj = torch.load(args.concept_cache, map_location="cpu")
    cc = obj.get("concept_logits", obj.get("concepts", obj))
    assert isinstance(cc, torch.Tensor), "concept_cache must contain a tensor under 'concept_logits' or 'concepts'"

    # meta + training mask
    concept_used, channel_used, use_mask_full = load_meta(args.meta, model)

    # image for this i
    hparams0 = hparams_registry.default_hparams("ERM", "Waterbirds")
    dset = vars(datasets)["Waterbirds"](args.data_dir, args.split, hparams0)
    x, y, a = find_example_by_i(dset, args.i)

    # concepts: logits + mean
    c_full = cc[args.i].float().cpu().numpy()              # [K_full]
    mu_full = cc.float().mean(dim=0).cpu().numpy()         # [K_full]

    c_used = c_full[use_mask_full]                         # [K_used]
    mu_used = mu_full[use_mask_full]                       # [K_used]

    # concept weights slice
    W = lin.weight.detach().cpu().numpy()
    w = W[args.class_idx]
    w_c = w[d : d + K_used]

    contrib_c = w_c * c_used                   # [K_used]

    # group by channel
    chan2idx = defaultdict(list)
    for j, ch in enumerate(channel_used):
        chan2idx[ch].append(j)

    wanted = [args.ch1, args.ch2]
    for ch in wanted:
        if ch not in chan2idx:
            raise ValueError(
                f"Channel '{ch}' not present after training mask. "
                f"Available: {sorted(list(chan2idx.keys()))}"
            )

    # prep channel panels
    panels = []
    for ch in wanted:
        idxs = np.array(chan2idx[ch], dtype=int)
        vals = contrib_c[idxs]
        labs = [concept_used[j] for j in idxs]

        pick = topk_abs(vals, k=min(args.topk, len(idxs)))
        vals_pick = vals[pick]
        labs_pick = [trunc(labs[j], 18) for j in pick]
        panels.append((ch, labs_pick, vals_pick))

    # ----------------
    # Plot: 3 x 1 vertical layout
    # ----------------
    fig = plt.figure(figsize=(3.2, 6.0), constrained_layout=False)

    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1.1, 1.0, 1.0],
        hspace=0.45
    )

    # --- IMAGE (row 0) shifted left
    ax_img = fig.add_axes([0.30, 0.69, 0.42, 0.26])
    #        [left, bottom, width, height]

    img = x.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    ax_img.imshow(img)
    ax_img.axis("off")


    # --- Morphology (row 1)
    ax_morph = fig.add_subplot(gs[1, 0])
    (chL, labsL, valsL) = panels[0]
    plot_channel(ax_morph, labsL, valsL, "Morphology", normalize=True)

    # --- Scene Context (row 2)
    ax_scene = fig.add_subplot(gs[2, 0])
    (chR, labsR, valsR) = panels[1]
    plot_channel(ax_scene, labsR, valsR, "Scene Context", normalize=True)

    # Margins so labels fit nicely
    fig.subplots_adjust(
        left=0.38,   # more space for long concept names
        right=0.95,
        top=0.98,
        bottom=0.08
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print("Saved ->", out)




if __name__ == "__main__":
    main()
