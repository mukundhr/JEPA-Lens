from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUT_FILE = BASE_DIR / "dashboard.html"


VARIANTS = [
    {
        "key": "baseline",
        "name": "Baseline v2",
        "accuracy": 51.4,
        "epochs": 60,
        "params": "3.1M",
        "description": "Standard JEPA with depth 6 and multi-crop masking.",
        "focus": "Reference model for all comparisons.",
        "takeaway": "Best probe score while keeping balanced structural and semantic sensitivity.",
        "ckpt": "baseline_v2",
        "files": {
            "umap": "understanding_map_baseline_v2.png",
            "tsne": "tsne_baseline_v2.png",
            "loss": "loss_curve_v2.png",
        },
    },
    {
        "key": "noise",
        "name": "Noise-Robust",
        "accuracy": 49.9,
        "epochs": 20,
        "params": "3.1M",
        "description": "Online encoder sees noisy input while the target stays clean.",
        "focus": "Pushes the model toward global context over brittle texture.",
        "takeaway": "Useful for testing whether the maps survive corruption pressure.",
        "ckpt": "noise_robust",
        "files": {
            "umap": "understanding_map_noise_robust.png",
            "tsne": "tsne_noise_robust.png",
            "loss": "loss_curve_noise_robust.png",
        },
    },
    {
        "key": "structure",
        "name": "Structure-Focused",
        "accuracy": 50.3,
        "epochs": 20,
        "params": "3.1M",
        "description": "Edge-biased input nudges the encoder toward contours and shape.",
        "focus": "Tests whether understanding maps are just edges or something richer.",
        "takeaway": "Separates structural bias from texture-heavy cues.",
        "ckpt": "structure_focused",
        "files": {
            "umap": "understanding_map_structure_focused.png",
            "tsne": "tsne_structure_focused.png",
            "loss": "loss_curve_structure_focused.png",
        },
    },
    {
        "key": "highmask",
        "name": "High-Mask",
        "accuracy": 49.1,
        "epochs": 20,
        "params": "3.1M",
        "description": "Masks 75 percent of patches so the encoder must infer more from less.",
        "focus": "Encourages broader context aggregation and coarser scene reasoning.",
        "takeaway": "Shows how stronger bottlenecks reshape the notion of complexity.",
        "ckpt": "high_mask",
        "files": {
            "umap": "understanding_map_high_mask.png",
            "tsne": "tsne_high_mask.png",
            "loss": "loss_curve_high_mask.png",
        },
    },
]

EXPERIMENTS = [
    {
        "title": "Masking Visualization",
        "subtitle": "What the encoder sees versus what it must predict.",
        "filename": "masking_viz_baseline.png",
        "command": "python visuals.py --checkpoint checkpoints/baseline_v2.pth",
        "alt": "Masking visualization for the baseline model.",
    },
    {
        "title": "Signal Nature Test",
        "subtitle": "Checks whether the maps collapse into raw edge detection.",
        "filename": "signal_nature_test.png",
        "command": "python signal_nature_test.py --baseline checkpoints/baseline_v2.pth --structure checkpoints/structure_focused.pth",
        "alt": "Signal nature test results.",
    },
    {
        "title": "Signal Nature Qualitative",
        "subtitle": "Qualitative companion figure for the signal test.",
        "filename": "signal_nature_qualitative.png",
        "command": "python signal_nature_test.py --baseline checkpoints/baseline_v2.pth --structure checkpoints/structure_focused.pth",
        "alt": "Signal nature qualitative results.",
    },
    {
        "title": "Transform Consistency",
        "subtitle": "Stability of understanding maps under flips and crops.",
        "filename": "transform_consistency.png",
        "command": "python consistency_test.py",
        "alt": "Transform consistency experiment.",
    },
    {
        "title": "Causal Test",
        "subtitle": "Ablates high-error patches to test whether they matter more.",
        "filename": "causal_test_baseline_v2.png",
        "command": "python causal_test.py --checkpoint checkpoints/baseline_v2.pth",
        "alt": "Causal test results for baseline v2.",
    },
]

ARTIFACTS = [
    ("Baseline map", "understanding_map_baseline_v2.png", "python evaluate.py --checkpoint checkpoints/baseline_v2.pth"),
    ("Noise map", "understanding_map_noise_robust.png", "python evaluate.py --checkpoint checkpoints/noise_robust.pth"),
    ("Structure map", "understanding_map_structure_focused.png", "python evaluate.py --checkpoint checkpoints/structure_focused.pth"),
    ("High-mask map", "understanding_map_high_mask.png", "python evaluate.py --checkpoint checkpoints/high_mask.pth"),
    ("Baseline t-SNE", "tsne_baseline_v2.png", "python evaluate.py --checkpoint checkpoints/baseline_v2.pth"),
    ("Noise t-SNE", "tsne_noise_robust.png", "python evaluate.py --checkpoint checkpoints/noise_robust.pth"),
    ("Structure t-SNE", "tsne_structure_focused.png", "python evaluate.py --checkpoint checkpoints/structure_focused.pth"),
    ("High-mask t-SNE", "tsne_high_mask.png", "python evaluate.py --checkpoint checkpoints/high_mask.pth"),
    ("Baseline loss", "loss_curve_v2.png", "python train.py"),
    ("Noise loss", "loss_curve_noise_robust.png", "python train_variants.py"),
    ("Structure loss", "loss_curve_structure_focused.png", "python train_variants.py"),
    ("High-mask loss", "loss_curve_high_mask.png", "python train_variants.py"),
    ("Masking visualization", "masking_viz_baseline.png", "python visuals.py --checkpoint checkpoints/baseline_v2.pth"),
    ("Signal nature test", "signal_nature_test.png", "python signal_nature_test.py --baseline checkpoints/baseline_v2.pth --structure checkpoints/structure_focused.pth"),
    ("Signal qualitative", "signal_nature_qualitative.png", "python signal_nature_test.py --baseline checkpoints/baseline_v2.pth --structure checkpoints/structure_focused.pth"),
    ("Transform consistency", "transform_consistency.png", "python consistency_test.py"),
    ("Causal test", "causal_test_baseline_v2.png", "python causal_test.py --checkpoint checkpoints/baseline_v2.pth"),
]

COMMANDS = [
    ("Train baseline", "python train.py"),
    ("Train variants", "python train_variants.py"),
    ("Evaluate a checkpoint", "python evaluate.py --checkpoint checkpoints/baseline_v2.pth"),
    ("Generate masking viz", "python visuals.py --checkpoint checkpoints/baseline_v2.pth"),
    ("Run validation suite", "python signal_nature_test.py --baseline checkpoints/baseline_v2.pth --structure checkpoints/structure_focused.pth"),
    ("Rebuild dashboard", "python dashboard.py"),
]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def img_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def status_badge(present: bool) -> str:
    tone = "ok" if present else "missing"
    label = "Present" if present else "Missing"
    return f'<span class="badge {tone}">{label}</span>'


def media_card(title: str, subtitle: str, src: str | None, alt: str, hint: str, tag: str = "") -> str:
    tag_html = f'<span class="tag">{esc(tag)}</span>' if tag else ""
    media = (
        f'<img src="{src}" alt="{esc(alt)}" loading="lazy">'
        if src
        else (
            '<div class="empty"><strong>Artifact missing</strong>'
            f"<p>{esc(subtitle)}</p><code>{esc(hint)}</code></div>"
        )
    )
    return (
        '<article class="card media">'
        '<div class="card-head">'
        f'<div><h3>{esc(title)}</h3><p>{esc(subtitle)}</p></div>'
        f'<div class="head-meta">{tag_html}{status_badge(bool(src))}</div>'
        "</div>"
        f'<div class="media-body">{media}</div>'
        "</article>"
    )


def accuracy_rows() -> str:
    rows = []
    for variant in VARIANTS:
        rows.append(
            '<div class="acc-row">'
            f'<span>{esc(variant["name"])}</span>'
            f'<div class="bar"><i style="width:{variant["accuracy"]:.1f}%"></i></div>'
            f'<strong>{variant["accuracy"]:.1f}%</strong>'
            "</div>"
        )
    rows.append(
        '<div class="acc-row muted"><span>Random baseline</span><div class="bar"><i style="width:10%"></i></div><strong>10.0%</strong></div>'
    )
    return "".join(rows)


def build_dashboard() -> str:
    variants = []
    for variant in VARIANTS:
        item = dict(variant)
        for slot, filename in variant["files"].items():
            item[f"{slot}_src"] = img_uri(OUTPUT_DIR / filename)
        variants.append(item)

    experiments = []
    for experiment in EXPERIMENTS:
        item = dict(experiment)
        item["src"] = img_uri(OUTPUT_DIR / experiment["filename"])
        experiments.append(item)

    variant_data = {
        variant["key"]: {
            "name": variant["name"],
            "accuracy": f'{variant["accuracy"]:.1f}%',
            "epochs": str(variant["epochs"]),
            "params": variant["params"],
            "description": variant["description"],
            "focus": variant["focus"],
            "takeaway": variant["takeaway"],
            "umap": variant["umap_src"] or "",
            "hint": f'python evaluate.py --checkpoint checkpoints/{variant["ckpt"]}.pth',
        }
        for variant in variants
    }

    present_count = sum(1 for _, filename, _ in ARTIFACTS if (OUTPUT_DIR / filename).exists())
    total_artifacts = len(ARTIFACTS)
    availability = (present_count / total_artifacts) * 100 if total_artifacts else 0.0
    best = max(variants, key=lambda item: item["accuracy"])
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    first = variants[0]

    variant_buttons = "".join(
        f'<button class="variant-btn{" active" if index == 0 else ""}" type="button" data-variant="{esc(variant["key"])}"><strong>{esc(variant["name"])}</strong><span>{variant["accuracy"]:.1f}% probe</span><small>{esc(variant["focus"])}</small></button>'
        for index, variant in enumerate(variants)
    )
    map_gallery = "".join(
        media_card(
            variant["name"],
            variant["takeaway"],
            variant["umap_src"],
            f'{variant["name"]} understanding map.',
            f'python evaluate.py --checkpoint checkpoints/{variant["ckpt"]}.pth',
            f'{variant["accuracy"]:.1f}% probe',
        )
        for variant in variants
    )
    tsne_gallery = "".join(
        media_card(
            f'{variant["name"]} t-SNE',
            "Embedding layout after self-supervised pretraining.",
            variant["tsne_src"],
            f'{variant["name"]} t-SNE embedding.',
            f'python evaluate.py --checkpoint checkpoints/{variant["ckpt"]}.pth',
            f'{variant["epochs"]} epochs',
        )
        for variant in variants
    )
    loss_gallery = "".join(
        media_card(
            f'{variant["name"]} loss curve',
            "Training stability and convergence for this run.",
            variant["loss_src"],
            f'{variant["name"]} training loss.',
            "python train.py" if variant["key"] == "baseline" else "python train_variants.py",
            f'{variant["epochs"]} epochs',
        )
        for variant in variants
    )
    experiment_gallery = "".join(
        media_card(
            experiment["title"],
            experiment["subtitle"],
            experiment["src"],
            experiment["alt"],
            experiment["command"],
        )
        for experiment in experiments
    )
    artifact_cards = "".join(
        f'<article class="card artifact {"present" if (OUTPUT_DIR / filename).exists() else "missing"}"><div class="card-head"><h3>{esc(label)}</h3>{status_badge((OUTPUT_DIR / filename).exists())}</div><p>outputs/{esc(filename)}</p><code>{esc(command)}</code></article>'
        for label, filename, command in ARTIFACTS
    )
    command_cards = "".join(
        f'<article class="card command"><h3>{esc(title)}</h3><code>{esc(command)}</code></article>'
        for title, command in COMMANDS
    )
    variant_json = json.dumps(variant_data, separators=(",", ":"))

    style = """
<style>
  :root {
    --bg: #f4eee2;
    --panel: rgba(255, 250, 242, 0.86);
    --card: rgba(255, 255, 255, 0.8);
    --ink: #1f1a17;
    --muted: #6e645b;
    --line: rgba(31, 26, 23, 0.12);
    --orange: #c25a2c;
    --orange-soft: rgba(194, 90, 44, 0.1);
    --teal: #0f6a66;
    --teal-soft: rgba(15, 106, 102, 0.1);
    --shadow: 0 16px 40px rgba(73, 48, 16, 0.1);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background:
      radial-gradient(circle at top left, rgba(15,106,102,0.12), transparent 26rem),
      radial-gradient(circle at right 20%, rgba(194,90,44,0.14), transparent 24rem),
      linear-gradient(180deg, #f7f1e7, #efe7da);
    color: var(--ink);
    font-family: "Space Grotesk", sans-serif;
    line-height: 1.5;
    padding: 24px 16px 40px;
  }
  body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
      linear-gradient(rgba(31, 26, 23, 0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(31, 26, 23, 0.04) 1px, transparent 1px);
    background-size: 30px 30px;
    mask-image: linear-gradient(180deg, rgba(0,0,0,0.45), transparent 82%);
  }
  .wrap { max-width: 1380px; margin: 0 auto; position: relative; z-index: 1; }
  .hero, .section { border: 1px solid var(--line); border-radius: 28px; background: var(--panel); box-shadow: var(--shadow); }
  .hero { padding: 28px; display: grid; gap: 24px; }
  .eyebrow { display: inline-flex; width: fit-content; gap: 10px; align-items: center; padding: 8px 12px; border-radius: 999px; background: var(--orange-soft); color: var(--orange); font: 0.78rem "IBM Plex Mono", monospace; text-transform: uppercase; }
  .hero-grid { display: grid; gap: 24px; grid-template-columns: minmax(0, 1fr); }
  .hero h1 { font-size: clamp(2.8rem, 6vw, 5.3rem); line-height: 0.95; letter-spacing: -0.05em; max-width: 11ch; }
  .hero h1 span { color: var(--orange); }
  .hero p.lead { max-width: 68ch; color: var(--muted); font-size: 1rem; margin-top: 16px; }
  .pill-row, .viewer-meta { display: flex; flex-wrap: wrap; gap: 10px; }
  .pill-row span, .viewer-meta span, .tag, .badge { padding: 8px 12px; border-radius: 999px; border: 1px solid var(--line); background: rgba(255,255,255,0.72); font: 0.78rem "IBM Plex Mono", monospace; }
  .callout { border-radius: 20px; padding: 18px; background: var(--teal-soft); border: 1px solid rgba(15,106,102,0.18); }
  .callout strong { display: block; font-size: 1.05rem; margin-bottom: 8px; }
  .callout p { color: var(--muted); }
  .stats { display: grid; gap: 16px; grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .stat { padding: 18px; border-radius: 20px; border: 1px solid var(--line); background: var(--card); }
  .stat small { display: block; margin-bottom: 10px; color: var(--muted); font: 0.78rem "IBM Plex Mono", monospace; text-transform: uppercase; }
  .stat strong { display: block; font-size: 1.9rem; letter-spacing: -0.05em; }
  .stat span { display: block; margin-top: 8px; color: var(--muted); font-size: 0.92rem; }
  .tabs { position: sticky; top: 12px; z-index: 5; margin: 20px 0 24px; display: flex; flex-wrap: wrap; gap: 10px; padding: 12px; border-radius: 999px; border: 1px solid var(--line); background: rgba(255,248,238,0.84); backdrop-filter: blur(16px); }
  .tabs button { border: 1px solid transparent; border-radius: 999px; background: transparent; color: var(--muted); padding: 10px 15px; font: inherit; cursor: pointer; }
  .tabs button.active { background: var(--ink); color: #fff8ef; border-color: var(--ink); }
  .pane { display: none; gap: 20px; }
  .pane.active { display: grid; }
  .section { padding: 24px; }
  .section-head { display: grid; gap: 10px; }
  .section-head small { color: var(--orange); font: 0.8rem "IBM Plex Mono", monospace; text-transform: uppercase; }
  .section-head h2 { font-size: clamp(1.9rem, 3vw, 2.8rem); letter-spacing: -0.04em; }
  .section-head p { max-width: 72ch; color: var(--muted); }
  .grid3, .gallery, .artifact-grid, .command-grid { display: grid; gap: 16px; }
  .grid3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .gallery, .artifact-grid, .command-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .mini, .step, .card { border-radius: 22px; border: 1px solid var(--line); background: var(--card); }
  .mini { padding: 18px; }
  .mini small { display: inline-flex; padding: 6px 10px; border-radius: 999px; margin-bottom: 10px; background: rgba(177,124,24,0.12); color: #a26f0a; font: 0.76rem "IBM Plex Mono", monospace; text-transform: uppercase; }
  .mini h3, .step h3, .card-head h3 { margin-bottom: 8px; font-size: 1.02rem; }
  .mini p, .step p, .card-head p, .artifact p, .acc-card p, .summary p { color: var(--muted); font-size: 0.93rem; }
  .steps { display: grid; gap: 14px; grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .step { padding: 18px; }
  .step span { display: inline-flex; width: 34px; height: 34px; align-items: center; justify-content: center; border-radius: 999px; background: var(--teal); color: white; font: 0.8rem "IBM Plex Mono", monospace; margin-bottom: 12px; }
  .split { display: grid; gap: 20px; grid-template-columns: 340px minmax(0, 1fr); align-items: start; }
  .variant-list { display: grid; gap: 12px; }
  .variant-btn { text-align: left; border: 1px solid var(--line); border-radius: 20px; background: var(--card); padding: 16px; cursor: pointer; color: inherit; }
  .variant-btn.active { background: var(--orange-soft); border-color: rgba(194,90,44,0.28); }
  .variant-btn strong { display: block; margin-bottom: 6px; font-size: 1rem; }
  .variant-btn span { display: block; color: var(--orange); font: 0.8rem "IBM Plex Mono", monospace; margin-bottom: 6px; }
  .variant-btn small { display: block; color: var(--muted); font-size: 0.9rem; }
  .viewer { display: grid; gap: 16px; }
  .figure { border-radius: 24px; overflow: hidden; border: 1px solid var(--line); background: var(--card); }
  .figure-head { display: flex; justify-content: space-between; gap: 12px; padding: 14px 16px; border-bottom: 1px solid var(--line); color: var(--muted); font: 0.82rem "IBM Plex Mono", monospace; }
  .figure img, .media-body img { display: block; width: 100%; height: auto; background: white; }
  .empty { min-height: 220px; display: grid; place-items: center; gap: 8px; text-align: center; padding: 24px; color: var(--muted); background: linear-gradient(135deg, rgba(15,106,102,0.08), transparent), rgba(255,255,255,0.82); }
  .empty strong { color: var(--ink); font-size: 1rem; }
  .card-head { display: flex; justify-content: space-between; align-items: start; gap: 12px; padding: 16px 16px 12px; border-bottom: 1px solid var(--line); }
  .head-meta { display: flex; flex-wrap: wrap; gap: 8px; justify-content: end; }
  .tag, .badge.ok { background: var(--teal-soft); color: var(--teal); border-color: rgba(15,106,102,0.16); }
  .badge.missing { background: var(--orange-soft); color: var(--orange); border-color: rgba(194,90,44,0.16); }
  .artifact, .command, .acc-card, .summary { padding: 16px; }
  .artifact p { margin-bottom: 12px; }
  code { display: inline-block; max-width: 100%; padding: 9px 11px; border-radius: 12px; border: 1px solid var(--line); background: rgba(255,255,255,0.84); font: 0.76rem "IBM Plex Mono", monospace; word-break: break-word; }
  .training { display: grid; gap: 20px; grid-template-columns: 330px minmax(0, 1fr); align-items: start; }
  .acc-card h3, .summary h3 { margin-bottom: 8px; font-size: 1.06rem; }
  .acc-row { display: grid; grid-template-columns: 116px minmax(0,1fr) 66px; gap: 10px; align-items: center; margin-bottom: 12px; }
  .acc-row span, .acc-row strong { font: 0.8rem "IBM Plex Mono", monospace; }
  .acc-row.muted { color: var(--muted); }
  .bar { height: 10px; border-radius: 999px; background: rgba(31,26,23,0.08); overflow: hidden; }
  .bar i { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--orange), #e19a48); }
  .summary-wrap { display: grid; gap: 20px; grid-template-columns: 320px minmax(0, 1fr); align-items: start; }
  .summary .big { font-size: 2.5rem; letter-spacing: -0.05em; margin-bottom: 10px; }
  .meter { height: 12px; border-radius: 999px; background: rgba(15,106,102,0.12); overflow: hidden; margin-bottom: 12px; }
  .meter i { display: block; height: 100%; background: linear-gradient(90deg, var(--teal), #2a9c92); }
  footer { margin-top: 24px; display: flex; flex-wrap: wrap; justify-content: space-between; gap: 12px; color: var(--muted); font-size: 0.9rem; padding: 0 6px; }
  footer a { color: var(--orange); text-decoration: none; }
  @media (max-width: 1100px) {
    .hero-grid, .split, .training, .summary-wrap { grid-template-columns: 1fr; }
    .stats, .grid3, .steps, .gallery, .artifact-grid, .command-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
  @media (max-width: 760px) {
    body { padding: 16px 10px 30px; }
    .hero, .section { padding: 18px; border-radius: 22px; }
    .stats, .grid3, .steps, .gallery, .artifact-grid, .command-grid { grid-template-columns: 1fr; }
    .figure-head, .card-head { flex-direction: column; align-items: start; }
    .acc-row { grid-template-columns: 1fr; }
  }
</style>
"""

    body = f"""
<div class="wrap">
  <header class="hero">
    <div class="eyebrow">JEPA-Lens / self-supervised report</div>
    <div class="hero-grid">
      <div>
        <h1>Mapping what a model <span>finds predictable</span></h1>
        <p class="lead">
          JEPA-Lens predicts masked patch representations instead of pixels and uses
          prediction error as a lens into model understanding. This dashboard now pulls
          the core maps, embeddings, validation figures, and training outputs into one
          synced static report.
        </p>
        <div class="pill-row" style="margin-top:18px">
          <span>32x32 CIFAR-10</span>
          <span>0 labels during pretraining</span>
          <span>Representation-space MSE</span>
          <span>4 controlled variants</span>
        </div>
      </div>
    </div>
    <div class="stats">
      <article class="stat"><small>Variants</small><strong>{len(variants)}</strong><span>Baseline plus three fine-tuned controls.</span></article>
      <article class="stat"><small>Artifact coverage</small><strong>{present_count}/{total_artifacts}</strong><span>{availability:.0f}% of tracked figures found in outputs/.</span></article>
      <article class="stat"><small>Best probe</small><strong>{best["accuracy"]:.1f}%</strong><span>{esc(best["name"])} is the top linear probe model.</span></article>
      <article class="stat"><small>Generated</small><strong>{esc(stamp)}</strong><span>Writes both dashboard HTML files in sync.</span></article>
    </div>
  </header>

  <nav class="tabs" aria-label="Dashboard sections">
    <button class="active" type="button" data-tab="overview">Overview</button>
    <button type="button" data-tab="understanding">Understanding</button>
    <button type="button" data-tab="representations">Representations</button>
    <button type="button" data-tab="validation">Validation</button>
    <button type="button" data-tab="training">Training</button>
    <button type="button" data-tab="artifacts">Artifacts</button>
  </nav>

  <main>
    <section class="pane active" id="pane-overview">
      <div class="section">
        <div class="section-head">
          <small>Overview</small>
          <h2>What the project is trying to prove</h2>
          <p>
            JEPA-Lens treats predictability as a measurement tool. If certain regions stay hard
            to infer from context, those regions may carry structure the model considers important.
          </p>
        </div>
      </div>
      <div class="section">
        <div class="grid3">
          <article class="mini"><small>Core question</small><h3>Which regions remain hard to predict?</h3><p>Low error marks easy context completion. High error marks uncertainty, ambiguity, or complex structure.</p></article>
          <article class="mini"><small>Main finding</small><h3>Prediction error is mixed, not trivial</h3><p>The validation figures suggest a stable structural component plus a finer texture-sensitive component.</p></article>
          <article class="mini"><small>Why variants matter</small><h3>The objective can steer complexity</h3><p>Noise, edge bias, and stronger masking shift what the model treats as difficult without destroying the representation.</p></article>
        </div>
      </div>
      <div class="section">
        <div class="section-head">
          <small>Pipeline</small>
          <h2>How the report maps to the workflow</h2>
          <p>Each phase creates a figure family that feeds directly into the dashboard.</p>
        </div>
        <div class="steps">
          <article class="step"><span>01</span><h3>Pretrain baseline</h3><p>Learn a context-predictive representation with the standard JEPA recipe.</p></article>
          <article class="step"><span>02</span><h3>Steer the objective</h3><p>Fine-tune controlled variants that alter noise, edges, and masking pressure.</p></article>
          <article class="step"><span>03</span><h3>Evaluate structure</h3><p>Generate maps, embeddings, masking views, and loss curves for every run.</p></article>
          <article class="step"><span>04</span><h3>Stress-test the claim</h3><p>Use signal, consistency, and causal tests to see whether the maps mean anything.</p></article>
        </div>
      </div>
    </section>

    <section class="pane" id="pane-understanding">
      <div class="section">
        <div class="section-head">
          <small>Understanding Maps</small>
          <h2>Interactive view of per-patch prediction error</h2>
          <p>
            Use the model rail to switch the main map, then scan the grid below for all four
            variants. Red patches are harder predictions, not object labels.
          </p>
        </div>
      </div>
      <div class="section">
        <div class="split">
          <div class="variant-list">{variant_buttons}</div>
          <div class="viewer">
            <div>
              <h3 id="viewer-title">{esc(first["name"])}</h3>
              <p style="color:var(--muted); margin-top:8px" id="viewer-description">{esc(first["description"])}</p>
            </div>
            <div class="viewer-meta" id="viewer-meta">
              <span>{first["accuracy"]:.1f}% linear probe</span>
              <span>{esc(first["params"])} params</span>
              <span>{esc(first["epochs"])} epochs</span>
            </div>
            <div class="figure">
              <div class="figure-head">
                <span id="viewer-caption">{esc(first["name"])}</span>
                <span>Red = harder to predict, green = easier to predict</span>
              </div>
              <div id="viewer-media">{'<img src="' + (first["umap_src"] or '') + '" alt="Understanding map">' if first["umap_src"] else '<div class="empty"><strong>Artifact missing</strong><p>Run the matching evaluation command to generate the map.</p><code>python evaluate.py --checkpoint checkpoints/baseline_v2.pth</code></div>'}</div>
            </div>
            <div class="callout">
              <strong id="viewer-focus">{esc(first["focus"])}</strong>
              <p id="viewer-takeaway">{esc(first["takeaway"])}</p>
            </div>
          </div>
        </div>
      </div>
      <div class="gallery">{map_gallery}</div>
    </section>

    <section class="pane" id="pane-representations">
      <div class="section">
        <div class="section-head">
          <small>Representations</small>
          <h2>Embedding structure and masking behavior</h2>
          <p>
            The t-SNE figures show how structure organizes in representation space, while the
            masking view grounds the heatmaps in the actual visible context versus hidden targets.
          </p>
        </div>
      </div>
      <div class="gallery">
        {media_card(experiments[0]["title"], experiments[0]["subtitle"], experiments[0]["src"], experiments[0]["alt"], experiments[0]["command"], "Baseline visual")}
        {tsne_gallery}
      </div>
    </section>

    <section class="pane" id="pane-validation">
      <div class="section">
        <div class="section-head">
          <small>Validation</small>
          <h2>Evidence that the maps are not just decoration</h2>
          <p>
            These figures test whether the predictability signal is stable, distinguishable from
            edge maps, and causally related to downstream semantics.
          </p>
        </div>
      </div>
      <div class="section">
        <div class="grid3">
          <article class="mini"><small>Signal nature</small><h3>Not raw edges</h3><p>Edge-focused training changes the maps, but the baseline signal does not collapse into a plain contour detector.</p></article>
          <article class="mini"><small>Consistency</small><h3>Partly stable under transforms</h3><p>Flips and crops preserve a structural component, suggesting the signal is not random patch noise.</p></article>
          <article class="mini"><small>Causality</small><h3>Hard patches matter more</h3><p>Ablating high-error regions should hurt probe accuracy more than removing random or easy patches.</p></article>
        </div>
      </div>
      <div class="gallery">{experiment_gallery}</div>
    </section>

    <section class="pane" id="pane-training">
      <div class="section">
        <div class="section-head">
          <small>Training</small>
          <h2>Probe summary and convergence curves</h2>
          <p>
            The probe scores stay close across variants, which is part of the point: the objective
            changes representation character more than overall viability.
          </p>
        </div>
      </div>
      <div class="section">
        <div class="training">
          <aside class="card acc-card">
            <h3>Linear probe snapshot</h3>
            <p>All four models stay far above the 10 percent random baseline.</p>
            {accuracy_rows()}
          </aside>
          <div class="gallery">{loss_gallery}</div>
        </div>
      </div>
    </section>

    <section class="pane" id="pane-artifacts">
      <div class="section">
        <div class="section-head">
          <small>Artifacts</small>
          <h2>What is present and how to regenerate it</h2>
          <p>
            If a figure is missing, the dashboard shows the command that should recreate it from
            the local project outputs.
          </p>
        </div>
      </div>
      <div class="section">
        <div class="summary-wrap">
          <aside class="card summary">
            <h3>Artifact health</h3>
            <p>The dashboard tracks the core training, evaluation, and validation figures.</p>
            <div class="big">{present_count}/{total_artifacts}</div>
            <div class="meter"><i style="width:{availability:.1f}%"></i></div>
            <p>{availability:.1f}% of tracked assets are available in <code>outputs/</code>.</p>
          </aside>
          <div class="artifact-grid">{artifact_cards}</div>
        </div>
      </div>
      <div class="section">
        <div class="section-head">
          <small>Commands</small>
          <h2>Fast path to rebuild the report</h2>
          <p>These are the commands you will actually use when refreshing results.</p>
        </div>
        <div class="command-grid">{command_cards}</div>
      </div>
    </section>
  </main>

  <footer>
    <p>Generated by dashboard.py on {esc(stamp)}. Writes dashboard.html.</p>
    <p><a href="https://arxiv.org/abs/2301.08243" target="_blank" rel="noreferrer">Assran et al. 2023</a> | <a href="https://github.com/mukundhr/JEPA-Lens" target="_blank" rel="noreferrer">Project repository</a></p>
  </footer>
</div>
"""

    script = f"""
<script id="variant-data" type="application/json">{variant_json}</script>
<script>
  const panes = Array.from(document.querySelectorAll(".pane"));
  const tabButtons = Array.from(document.querySelectorAll("[data-tab]"));
  function activateTab(name) {{
    panes.forEach((pane) => pane.classList.toggle("active", pane.id === "pane-" + name));
    tabButtons.forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  }}
  tabButtons.forEach((button) => button.addEventListener("click", () => activateTab(button.dataset.tab)));

  const variants = JSON.parse(document.getElementById("variant-data").textContent);
  const variantButtons = Array.from(document.querySelectorAll("[data-variant]"));
  const viewerTitle = document.getElementById("viewer-title");
  const viewerDescription = document.getElementById("viewer-description");
  const viewerCaption = document.getElementById("viewer-caption");
  const viewerFocus = document.getElementById("viewer-focus");
  const viewerTakeaway = document.getElementById("viewer-takeaway");
  const viewerMeta = document.getElementById("viewer-meta");
  const viewerMedia = document.getElementById("viewer-media");

  function renderViewer(variant) {{
    viewerTitle.textContent = variant.name;
    viewerDescription.textContent = variant.description;
    viewerCaption.textContent = variant.name;
    viewerFocus.textContent = variant.focus;
    viewerTakeaway.textContent = variant.takeaway;
    viewerMeta.innerHTML =
      "<span>" + variant.accuracy + " linear probe</span>" +
      "<span>" + variant.params + " params</span>" +
      "<span>" + variant.epochs + " epochs</span>";
    if (variant.umap) {{
      viewerMedia.innerHTML = '<img src="' + variant.umap + '" alt="' + variant.name + ' understanding map">';
    }} else {{
      viewerMedia.innerHTML = '<div class="empty"><strong>Artifact missing</strong><p>Run the matching evaluation command to generate the map.</p><code>' + variant.hint + '</code></div>';
    }}
  }}

  function activateVariant(key) {{
    const variant = variants[key];
    if (!variant) return;
    variantButtons.forEach((button) => button.classList.toggle("active", button.dataset.variant === key));
    renderViewer(variant);
  }}

  variantButtons.forEach((button) => button.addEventListener("click", () => activateVariant(button.dataset.variant)));
  activateTab("overview");
  activateVariant("{esc(first["key"])}");
</script>
"""

    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">'
        "<title>JEPA-Lens Dashboard</title>"
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">'
        + style
        + "</head><body>"
        + body
        + script
        + "</body></html>"
    )


def main() -> None:
    html_text = build_dashboard()
    OUT_FILE.write_text(html_text, encoding="utf-8")
    size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Dashboard written to {OUT_FILE.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
