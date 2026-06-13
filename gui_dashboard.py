#!/usr/bin/env python3
"""
SC-ViT Research Dashboard — Gradio GUI.

Tabs:
  1. Experiment Results — metrics table + comparison charts
  2. Visualization Gallery — browse all generated images
  3. Test on Images — run inference on 1-5 dataset images

Usage:
    python gui_dashboard.py
"""

import os, sys, json, glob
import numpy as np
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(BASE, "outputs")
EXPERIMENTS = os.path.join(OUTPUTS, "experiments")
VISUALIZATIONS = os.path.join(OUTPUTS, "visualizations")
CACHE = os.path.join(OUTPUTS, "cache")

TARGET_NAMES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


# ════════════════════════════════════════════════════════════
#  Tab 1: Experiment Results
# ════════════════════════════════════════════════════════════

def load_results():
    """Load experiment results from JSON."""
    for path in [
        os.path.join(EXPERIMENTS, "all_results.json"),
        os.path.join(OUTPUTS, "results", "experiment_summary.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f), path
    return None, None


def build_results_table():
    """Build markdown table from results."""
    data, path = load_results()
    if not data:
        return "No experiment results found. Run experiments first.", ""

    header = "| Model | AUROC | F1 | Accuracy | Entropy | Time (min) |"
    sep = "|:------|------:|---:|---------:|--------:|-----------:|"
    rows = [header, sep]

    for r in data:
        auroc = r.get("auroc_mean", r.get("auroc", {}).get("mean", 0))
        f1 = r.get("f1_mean", r.get("f1", {}).get("mean", 0))
        acc = r.get("accuracy_mean", r.get("accuracy", {}).get("mean", 0))
        ent = r.get("attention_entropy", 0)
        if isinstance(ent, dict):
            ent = ent.get("mean", 0)
        tm = r.get("train_time_min", 0)
        name = r.get("model_name", "Unknown")
        rows.append(f"| {name} | {auroc:.4f} | {f1:.4f} | {acc:.4f} | {ent:.3f} | {tm:.1f} |")

    # Per-class AUROC if available
    has_per_class = any("per_class_auroc" in r for r in data)
    if has_per_class:
        rows.append("")
        rows.append("### Per-Class AUROC")
        rows.append("| Model | " + " | ".join(TARGET_NAMES) + " |")
        rows.append("|:------|" + "------:|" * len(TARGET_NAMES))
        for r in data:
            pc = r.get("per_class_auroc", {})
            vals = " | ".join(f"{pc.get(c, 0):.4f}" for c in TARGET_NAMES)
            rows.append(f"| {r.get('model_name', '?')} | {vals} |")

    return "\n".join(rows), f"Source: `{path}`"


def get_results_images():
    """Get experiment comparison chart images."""
    imgs = []
    search_dirs = [EXPERIMENTS, os.path.join(OUTPUTS, "results")]
    for d in search_dirs:
        for name in [
            "main_metrics_comparison.png",
            "per_class_auroc_heatmap.png",
            "attention_entropy.png",
            "attention_entropy_comparison.png",
            "noise_robustness.png",
            "noise_robustness_curves.png",
            "radar_chart.png",
            "anatomical_attention_ratio.png",
        ]:
            p = os.path.join(d, name)
            if os.path.exists(p):
                imgs.append(p)
    return imgs if imgs else None


# ════════════════════════════════════════════════════════════
#  Tab 2: Visualization Gallery
# ════════════════════════════════════════════════════════════

def collect_all_images():
    """Collect all PNG images organized by category."""
    categories = {}
    for root, dirs, files in os.walk(OUTPUTS):
        pngs = sorted([f for f in files if f.endswith(".png")])
        if not pngs:
            continue
        rel = os.path.relpath(root, OUTPUTS)
        cat = rel.split(os.sep)[0] if os.sep in rel else rel
        cat = cat.replace("_", " ").title()
        if cat == ".":
            cat = "Root"
        if cat not in categories:
            categories[cat] = []
        for p in pngs:
            full = os.path.join(root, p)
            categories[cat].append((full, p))
    return categories


def get_category_choices():
    cats = collect_all_images()
    return list(cats.keys()) if cats else ["No images found"]


def get_images_for_category(category):
    cats = collect_all_images()
    if category not in cats:
        return []
    return [path for path, name in cats[category]]


# ════════════════════════════════════════════════════════════
#  Tab 3: Test on Images
# ════════════════════════════════════════════════════════════

def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_for_inference(model_type="baseline"):
    """Load a trained model for inference, auto-detecting architecture from checkpoint."""
    import torch, yaml

    with open(os.path.join(BASE, "configs", "default.yaml")) as f:
        config = yaml.safe_load(f)

    device = get_device()

    # Determine checkpoint path
    ckpt_dirs = {
        "baseline": "Baseline_ViT",
        "kmeans": "ViT_plus_KMeans",
        "spectral": "ViT_plus_Spectral",
        "gnn": "ViT_plus_Graph_GNN",
        "constrained": "ViT_plus_Struct-Constrained",
        "hybrid_relational": None,  # uses new naming
    }

    ckpt_dir = ckpt_dirs.get(model_type)
    if ckpt_dir:
        ckpt_path = os.path.join(OUTPUTS, "experiments", "checkpoints", ckpt_dir, "best_model.pt")
    else:
        ckpt_path = ""

    if not os.path.exists(ckpt_path):
        tag = model_type.lower().replace(" ", "_")
        ckpt_path = os.path.join(OUTPUTS, "checkpoints", f"{tag}_best_model.pt")

    # Auto-detect architecture from checkpoint
    img_size = config["data"].get("image_size", 192)
    embed_dim = 768
    num_heads = 12
    depth = 12
    model_kind = "baseline"  # 'baseline', 'constrained', 'hybrid_relational'

    state = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        for key in ["pos_embed", "vit.pos_embed", "backbone.pos_embed"]:
            if key in state:
                _, n_tokens, d = state[key].shape
                embed_dim = d
                n_patches = n_tokens - 1
                grid = int(n_patches ** 0.5)
                img_size = grid * 16
                break

        if embed_dim == 384:
            num_heads = 6
        elif embed_dim == 192:
            num_heads = 3

        block_indices = set()
        for k in state.keys():
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    block_indices.add(int(parts[i + 1]))
        if block_indices:
            depth = max(block_indices) + 1

        # Detect model type from state_dict
        if "lambda_graph" in state:
            model_kind = "hybrid_relational"
        elif "lambda_structure" in state:
            model_kind = "constrained"

    # Build model matching checkpoint
    from src.models.vit import VisionTransformer
    if model_kind == "hybrid_relational":
        from src.models.hybrid_relational_vit import HybridAnatomyGuidedRelationalViT
        model = HybridAnatomyGuidedRelationalViT(img_size=img_size, num_classes=5,
                                                  embed_dim=embed_dim, num_heads=num_heads, depth=depth)
    elif model_kind == "constrained":
        from src.models.constrained_vit import ConstrainedVisionTransformer
        model = ConstrainedVisionTransformer(img_size=img_size, num_classes=5,
                                             embed_dim=embed_dim, num_heads=num_heads, depth=depth)
    else:
        model = VisionTransformer(img_size=img_size, num_classes=5,
                                  embed_dim=embed_dim, num_heads=num_heads, depth=depth)

    if state is not None:
        model.load_state_dict(state, strict=False)

    model = model.to(device)
    model.eval()
    return model, device, img_size, model_kind


def learn_temperature(model, dataset, device, F_mat, n_cal=500,
                      graph_bias=None, anatomy_prior=None, model_kind="baseline"):
    """
    Learn optimal temperature T for logit calibration via NLL grid search.

    Temperature scaling: scaled_probs = sigmoid(logits / T)
    Spreads compressed probabilities apart for better thresholding.
    """
    import torch
    from sklearn.metrics import log_loss

    model.eval()
    all_logits, all_targets = [], []

    n_cal = min(n_cal, len(dataset))
    for idx in range(n_cal):
        img, gt = dataset[idx]
        img_batch = img.unsqueeze(0).to(device)
        with torch.no_grad():
            if model_kind == "hybrid_relational":
                logits = model(img_batch, graph_bias=graph_bias, anatomy_prior=anatomy_prior)
            elif F_mat is not None:
                logits = model(img_batch, feasibility_matrix=F_mat)
            else:
                logits = model(img_batch)
            all_logits.append(logits.cpu().numpy()[0])
        all_targets.append(gt.numpy())

    all_logits = np.array(all_logits)
    all_targets = np.array(all_targets)

    best_T = 1.0
    best_nll = float('inf')
    for T in np.arange(0.3, 3.0, 0.05):
        scaled_probs = 1.0 / (1.0 + np.exp(-all_logits / T))
        scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)
        try:
            nll = 0
            count = 0
            for j in range(5):
                valid = ~np.isnan(all_targets[:, j])
                if valid.sum() > 0:
                    nll += log_loss(all_targets[valid, j], scaled_probs[valid, j])
                    count += 1
            if count > 0:
                nll /= count
                if nll < best_nll:
                    best_nll = nll
                    best_T = T
        except Exception:
            continue

    return best_T, all_logits, all_targets


def calibrate_thresholds_v2(all_logits, all_targets, temperature=1.0, margin=0.03):
    """
    Per-class thresholds via Fβ=0.5 sweep + temperature scaling + margin.

    Fixes over-prediction by:
    1. Temperature scaling spreads compressed probabilities
    2. Fβ=0.5 penalizes false positives 2x more than false negatives
    3. Confidence margin avoids weak positives
    4. Minimum threshold floor prevents absurdly low thresholds
    """
    from sklearn.metrics import fbeta_score

    scaled_probs = 1.0 / (1.0 + np.exp(-all_logits / temperature))

    thresholds = {}
    for j, name in enumerate(TARGET_NAMES):
        y_true = all_targets[:, j]
        y_prob = scaled_probs[:, j]

        valid = ~np.isnan(y_true)
        y_true_v = y_true[valid]
        y_prob_v = y_prob[valid]

        if len(np.unique(y_true_v)) < 2:
            thresholds[name] = 0.35
            continue

        best_score = 0.0
        best_t = 0.3
        for t in np.arange(0.10, 0.70, 0.01):
            preds = (y_prob_v > t).astype(int)
            if preds.sum() == 0:
                continue
            score = fbeta_score(y_true_v, preds, beta=0.5, zero_division=0)
            if score > best_score:
                best_score = score
                best_t = t

        best_t = max(best_t, 0.20)
        best_t += margin
        thresholds[name] = round(float(best_t), 3)

    return thresholds


def run_inference(num_images, model_type):
    """Run inference with temperature-scaled, Fβ=0.5 calibrated predictions."""
    import torch
    import yaml
    from src.data.transforms import get_val_transforms
    from src.data.chexpert_dataset import CheXpertDataset
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(BASE, "configs", "default.yaml")) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    dataset_dir = data_cfg["dataset_dir"]

    if not os.path.exists(dataset_dir):
        return None, "Dataset not found. Please check configs/default.yaml"

    try:
        model, device, img_size, model_kind = load_model_for_inference(model_type)
    except Exception as e:
        return None, f"Model load error: {e}"

    valid_csv = os.path.join(dataset_dir, data_cfg["valid_csv"])
    transform = get_val_transforms(img_size)

    dataset = CheXpertDataset(
        csv_path=valid_csv, dataset_dir=dataset_dir,
        target_columns=TARGET_NAMES, uncertainty_policy="ones",
        transform=transform, subset_size=500,
    )

    num_images = min(int(num_images), len(dataset), 5)

    # Load bias matrices
    F_mat, G_mat, A_mat = None, None, None
    n_tokens = (img_size // 16) ** 2 + 1

    if model_kind == "hybrid_relational":
        for fname, attr in [("F_feature.pt", "G"), ("F_anatomy.pt", "A")]:
            fpath = os.path.join(CACHE, fname)
            if os.path.exists(fpath):
                data = torch.load(fpath, map_location=device, weights_only=False)
                mat = data["feasibility_matrix"].to(device)
                if mat.shape[0] == n_tokens:
                    if attr == "G":
                        G_mat = mat
                    else:
                        A_mat = mat
        if A_mat is None:
            from src.structure.anatomy_prior import AnatomyPriorBuilder
            A_mat = AnatomyPriorBuilder().build(img_size // 16).to(device)
    elif model_kind == "constrained":
        for fname in ["F_hybrid.pt", "F_anatomy.pt"]:
            fpath = os.path.join(CACHE, fname)
            if os.path.exists(fpath):
                data = torch.load(fpath, map_location=device, weights_only=False)
                F_mat = data["feasibility_matrix"].to(device)
                if F_mat.shape[0] != n_tokens:
                    F_mat = None
                break

    # Step 1: Learn temperature on 500 images
    temperature, all_logits, all_targets = learn_temperature(
        model, dataset, device, F_mat, n_cal=500,
        graph_bias=G_mat, anatomy_prior=A_mat, model_kind=model_kind,
    )

    # Step 2: Calibrate thresholds with Fβ=0.5 + margin
    thresholds = calibrate_thresholds_v2(
        all_logits, all_targets, temperature=temperature, margin=0.03,
    )

    rng = np.random.default_rng()
    indices = rng.choice(len(dataset), size=num_images, replace=False)

    model_label = "Hybrid Relational" if model_kind == "hybrid_relational" else model_type.title()
    results_md = f"## Inference Results — {model_label} Model\n\n"
    results_md += f"### Calibration Settings\n"
    results_md += f"- **Temperature:** T = {temperature:.2f} *(learned on {len(dataset)} val images)*\n"
    results_md += f"- **Threshold metric:** Fβ=0.5 *(penalizes false positives 2× more)*\n"
    results_md += f"- **Confidence margin:** +0.03 *(avoids weak positives)*\n\n"
    results_md += "| Pathology | Threshold |\n|:----------|----------:|\n"
    for name in TARGET_NAMES:
        t = thresholds.get(name, 0.5)
        results_md += f"| {name} | **{t:.3f}** |\n"
    results_md += "\n---\n\n"

    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1:
        axes = [axes]

    total_correct = 0
    total_preds = 0

    for i, idx in enumerate(indices):
        img_tensor, gt_labels = dataset[idx]
        img_batch = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            if model_kind == "hybrid_relational":
                logits = model(img_batch, graph_bias=G_mat, anatomy_prior=A_mat)
            elif F_mat is not None:
                logits = model(img_batch, feasibility_matrix=F_mat)
            else:
                logits = model(img_batch)
            scaled_logits = logits / temperature
            probs = torch.sigmoid(scaled_logits).cpu().numpy()[0]

        gt = gt_labels.numpy()

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        axes[i].imshow(img_np)
        axes[i].set_title(f"Image {idx}", fontsize=11, fontweight="bold")
        axes[i].axis("off")

        results_md += f"### Image {idx}\n"
        results_md += "| Pathology | Ground Truth | Predicted | Prob | Threshold |\n"
        results_md += "|:----------|:-----------:|:---------:|-----:|----------:|\n"
        for j, name in enumerate(TARGET_NAMES):
            t = thresholds.get(name, 0.5)
            gt_positive = gt[j] > 0.5
            pred_positive = probs[j] > t
            gt_str = "✅ Yes" if gt_positive else "❌ No"
            pred_str = "✅ Yes" if pred_positive else "❌ No"
            match = "✓" if gt_positive == pred_positive else "✗"
            results_md += f"| {name} | {gt_str} | {pred_str} | {probs[j]:.3f} | {t:.3f} {match} |\n"
            if gt_positive == pred_positive:
                total_correct += 1
            total_preds += 1
        results_md += "\n"

    accuracy = total_correct / max(total_preds, 1)
    results_md += f"\n**Overall accuracy: {total_correct}/{total_preds} = {accuracy:.1%}**\n"

    plt.tight_layout()
    save_path = os.path.join(OUTPUTS, "visualizations", "inference_results.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return save_path, results_md


# ════════════════════════════════════════════════════════════
#  Build Gradio App
# ════════════════════════════════════════════════════════════

def build_app():
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(theme=theme, title="SC-ViT Research Dashboard") as app:
        gr.Markdown(
            "# 🔬 Structure-Constrained ViT — Research Dashboard\n"
            "Hybrid Anatomy-Aware Vision Transformer for Chest X-ray Classification"
        )

        # ── Tab 1: Results ──
        with gr.Tab("📊 Experiment Results"):
            gr.Markdown("### Model Comparison Table")
            table_md, source_md = build_results_table()
            gr.Markdown(table_md)
            if source_md:
                gr.Markdown(source_md)

            gr.Markdown("### Comparison Charts")
            chart_imgs = get_results_images()
            if chart_imgs:
                gr.Gallery(value=chart_imgs, columns=2, height=500, label="Experiment Charts")
            else:
                gr.Markdown("*No comparison charts found. Run experiments first.*")

        # ── Tab 2: Gallery ──
        with gr.Tab("🖼️ Visualization Gallery"):
            gr.Markdown("### Browse All Generated Visualizations")
            categories = collect_all_images()

            if categories:
                cat_dropdown = gr.Dropdown(
                    choices=list(categories.keys()),
                    value=list(categories.keys())[0],
                    label="Category",
                )
                gallery = gr.Gallery(
                    value=get_images_for_category(list(categories.keys())[0]),
                    columns=2, height=600, label="Images",
                )
                cat_dropdown.change(
                    fn=get_images_for_category,
                    inputs=cat_dropdown,
                    outputs=gallery,
                )
            else:
                gr.Markdown("*No images found in outputs/*")

        # ── Tab 3: Test on Images ──
        with gr.Tab("🧪 Test on Images"):
            gr.Markdown(
                "### Run Inference on Validation Images\n"
                "Select a model and number of images to test. "
                "The model will predict pathology probabilities for each image."
            )

            with gr.Row():
                model_choice = gr.Dropdown(
                    choices=["baseline", "kmeans", "spectral", "gnn", "constrained", "hybrid_relational"],
                    value="baseline",
                    label="Model",
                )
                num_imgs = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="Number of Images",
                )

            run_btn = gr.Button("🚀 Run Inference", variant="primary")

            with gr.Row():
                result_image = gr.Image(label="Input Images", type="filepath")
                result_text = gr.Markdown(label="Predictions")

            run_btn.click(
                fn=run_inference,
                inputs=[num_imgs, model_choice],
                outputs=[result_image, result_text],
            )

        # ── Tab 4: Architecture ──
        with gr.Tab("📐 Architecture"):
            gr.Markdown("""
### Pipeline Architecture

```
Image (192×192)
  → Patch Embedding (12×12 = 144 patches)
  → ViT Backbone (frozen first 70%)
  → [Precomputed] F_anatomy (145×145) — heuristic spatial prior
  → [Precomputed] F_feature (145×145) — DINO + spectral clustering
  → F_hybrid = 0.7·F_anatomy + 0.3·F_feature
  → Soft Structural Attention: softmax(QK^T/√d + λ·log(F_hybrid + ε))
  → Multi-label Classification (5 pathologies)
```

### Key Innovation
**Pre-softmax additive log-bias** (soft constraint) instead of post-softmax multiplicative mask (hard constraint):
- Old: `Attention = softmax(QK^T / √d) ⊙ A` — zeroes out interactions
- New: `Attention = softmax(QK^T / √d + λ·log(F + ε))` — nudges attention

### 7 Anatomical Regions
| Region | Description |
|--------|------------|
| 0 | Left Upper Lung |
| 1 | Left Lower Lung |
| 2 | Right Upper Lung |
| 3 | Right Lower Lung |
| 4 | Heart |
| 5 | Mediastinum |
| 6 | Background |
""")
            anat_img = os.path.join(VISUALIZATIONS, "anatomy_regions.png")
            if os.path.exists(anat_img):
                gr.Image(value=anat_img, label="Anatomy Region Map")

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
