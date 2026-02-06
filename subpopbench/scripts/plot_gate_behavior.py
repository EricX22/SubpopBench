import json, argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_meta(meta_path):
    meta = json.loads(Path(meta_path).read_text())
    # You likely have meta["concepts"] = list of dicts, or similar
    concepts = meta.get("concepts", meta)  # be tolerant
    return concepts

def categorize(name: str) -> str:
    n = name.lower()
    # Adjust rules to match how your concept bank names look
    if any(k in n for k in ["water", "land", "background", "scene", "indoor", "outdoor", "environment"]):
        return "Environment"
    if any(k in n for k in ["bird", "species", "smiling", "pathology", "pneum", "lesion", "disease", "label"]):
        return "Target attributes"
    if any(k in n for k in ["branch", "cage", "accessory", "object", "tree", "grass", "rock"]):
        return "Co-occurring objects"
    if any(k in n for k in ["border", "scanner", "artifact", "noise", "text", "mark"]):
        return "Artifacts"
    return "Other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to saved model checkpoint (or model.pkl if you torch.save)")
    ap.add_argument("--meta", required=True, help="concept meta json")
    ap.add_argument("--out", default="gate_behavior.png")
    args = ap.parse_args()

    concepts = load_meta(args.meta)
    names = []
    for c in concepts:
        if isinstance(c, dict):
            names.append(c.get("name", c.get("concept", "unknown")))
        else:
            names.append(str(c))

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # If you saved state_dict:
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Common patterns:
    # 1) gate_logits directly in state dict
    gate_key = None
    for k in state.keys():
        if k.endswith("gate_logits") or "gate_logits" in k:
            gate_key = k
            break
    if gate_key is None:
        raise RuntimeError("Could not find gate_logits in checkpoint keys. Print keys and adjust.")

    gate_logits = state[gate_key].detach().cpu().float()
    gates = torch.sigmoid(gate_logits).numpy()

    # Align lengths if necessary
    K = min(len(gates), len(names))
    gates = gates[:K]
    names = names[:K]

    cats = [categorize(n) for n in names]
    uniq = ["Target attributes", "Environment", "Co-occurring objects", "Artifacts", "Other"]

    # Aggregate
    cat_means = []
    cat_stds = []
    for u in uniq:
        vals = [g for g,c in zip(gates, cats) if c == u]
        if len(vals) == 0:
            cat_means.append(np.nan); cat_stds.append(np.nan)
        else:
            cat_means.append(float(np.mean(vals)))
            cat_stds.append(float(np.std(vals)))

    plt.figure()
    plt.bar(range(len(uniq)), cat_means, yerr=cat_stds)
    plt.xticks(range(len(uniq)), uniq, rotation=25, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Average gate value (sigmoid)")
    plt.title("Concept gate behavior by channel")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Wrote", args.out)
    print("gate_key:", gate_key)

if __name__ == "__main__":
    main()
