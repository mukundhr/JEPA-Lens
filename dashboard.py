"""
build_dashboard.py — Generate self-contained JEPA-Lens HTML dashboard
======================================================================
Reads all output images, converts to base64, bakes into a single HTML file.
Open jepa_lens_dashboard.html in any browser — no server needed.

Run:
  python build_dashboard.py
"""

import base64, os, json
from pathlib import Path

def img_to_b64(path):
    """Convert image file to base64 data URI."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

# ── LOAD ALL IMAGES ───────────────────────────────────────────────────────────
images = {
    # Loss curves
    "loss_baseline":          img_to_b64("outputs/loss_curve_v2.png"),
    "loss_noise":             img_to_b64("outputs/loss_curve_noise_robust.png"),
    "loss_structure":         img_to_b64("outputs/loss_curve_structure_focused.png"),
    "loss_highmask":          img_to_b64("outputs/loss_curve_high_mask.png"),

    # t-SNE
    "tsne_baseline":          img_to_b64("outputs/tsne_baseline_v2.png"),
    "tsne_noise":             img_to_b64("outputs/tsne_noise_robust.png"),
    "tsne_structure":         img_to_b64("outputs/tsne_structure_focused.png"),
    "tsne_highmask":          img_to_b64("outputs/tsne_high_mask.png"),

    # Understanding maps
    "umap_baseline":          img_to_b64("outputs/understanding_map_baseline_v2.png"),
    "umap_noise":             img_to_b64("outputs/understanding_map_noise_robust.png"),
    "umap_structure":         img_to_b64("outputs/understanding_map_structure_focused.png"),
    "umap_highmask":          img_to_b64("outputs/understanding_map_high_mask.png"),

    # Masking viz (baseline only)
    "masking":                img_to_b64("outputs/masking_viz_baseline.png"),
}

# Filter out missing images
missing = [k for k, v in images.items() if v is None]
if missing:
    print(f"Warning: missing images: {missing}")
    print("Run the relevant scripts first to generate them.")

# Stats
stats = {
    "baseline":   {"acc": 51.4, "params": "3.1M", "epochs": 60},
    "noise":      {"acc": 49.9, "params": "3.1M", "epochs": 20},
    "structure":  {"acc": 50.3, "params": "3.1M", "epochs": 20},
    "highmask":   {"acc": 49.1, "params": "3.1M", "epochs": 20},
}

# ── HTML ──────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JEPA-Lens</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:       #0a0a0f;
    --surface:  #111118;
    --border:   #1e1e2e;
    --accent:   #7c6aff;
    --accent2:  #ff6a6a;
    --accent3:  #6affb8;
    --text:     #e2e2f0;
    --muted:    #6b6b88;
    --card:     #13131c;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Mono', monospace;
    min-height: 100vh;
    overflow-x: hidden;
  }}

  /* ── GRID TEXTURE ── */
  body::before {{
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.3;
    pointer-events: none;
    z-index: 0;
  }}

  /* ── HERO ── */
  .hero {{
    position: relative;
    z-index: 1;
    padding: 80px 60px 60px;
    border-bottom: 1px solid var(--border);
  }}

  .hero-tag {{
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 20px;
  }}

  .hero h1 {{
    font-family: 'Syne', sans-serif;
    font-size: clamp(36px, 6vw, 72px);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -2px;
    margin-bottom: 20px;
  }}

  .hero h1 span {{
    color: var(--accent);
  }}

  .hero-sub {{
    font-size: 13px;
    color: var(--muted);
    max-width: 600px;
    line-height: 1.8;
    margin-bottom: 40px;
  }}

  .hero-stats {{
    display: flex;
    gap: 40px;
    flex-wrap: wrap;
  }}

  .stat {{
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}

  .stat-val {{
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--accent3);
  }}

  .stat-label {{
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--muted);
    text-transform: uppercase;
  }}

  /* ── NAV TABS ── */
  .nav {{
    position: relative;
    z-index: 1;
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    padding: 0 60px;
    overflow-x: auto;
  }}

  .tab {{
    padding: 16px 24px;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
    white-space: nowrap;
    background: none;
    border-top: none;
    border-left: none;
    border-right: none;
  }}

  .tab:hover {{ color: var(--text); }}
  .tab.active {{
    color: var(--accent);
    border-bottom-color: var(--accent);
  }}

  /* ── MAIN CONTENT ── */
  .main {{
    position: relative;
    z-index: 1;
    padding: 60px;
  }}

  .section {{ display: none; }}
  .section.active {{ display: block; }}

  /* ── SECTION HEADER ── */
  .section-header {{
    margin-bottom: 40px;
  }}

  .section-title {{
    font-family: 'Syne', sans-serif;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 8px;
  }}

  .section-desc {{
    font-size: 12px;
    color: var(--muted);
    line-height: 1.7;
    max-width: 700px;
  }}

  /* ── MODEL GRID ── */
  .model-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
  }}

  .model-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    transition: border-color 0.2s;
    cursor: pointer;
  }}

  .model-card:hover {{ border-color: var(--accent); }}
  .model-card.selected {{ border-color: var(--accent); }}

  .model-card-header {{
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  .model-name {{
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
  }}

  .model-acc {{
    font-size: 12px;
    color: var(--accent3);
  }}

  .model-desc {{
    padding: 12px 20px;
    font-size: 11px;
    color: var(--muted);
    line-height: 1.6;
  }}

  /* ── IMAGE DISPLAY ── */
  .img-container {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 24px;
  }}

  .img-header {{
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}

  .img-header span {{ color: var(--accent); }}

  .img-container img {{
    width: 100%;
    display: block;
  }}

  /* ── COMPARISON GRID ── */
  .compare-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
  }}

  @media (max-width: 900px) {{
    .compare-grid {{ grid-template-columns: 1fr; }}
    .hero {{ padding: 40px 24px 40px; }}
    .nav {{ padding: 0 24px; }}
    .main {{ padding: 40px 24px; }}
  }}

  /* ── ACCURACY BAR ── */
  .acc-bars {{
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin: 32px 0;
  }}

  .acc-row {{
    display: flex;
    align-items: center;
    gap: 16px;
  }}

  .acc-label {{
    font-size: 11px;
    width: 160px;
    color: var(--muted);
    flex-shrink: 0;
  }}

  .acc-bar-wrap {{
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }}

  .acc-bar-fill {{
    height: 100%;
    border-radius: 3px;
    background: var(--accent);
    transition: width 1s ease;
  }}

  .acc-val {{
    font-size: 12px;
    color: var(--accent3);
    width: 50px;
    text-align: right;
    flex-shrink: 0;
  }}

  /* ── KEY FINDING BOX ── */
  .finding {{
    background: var(--card);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 24px 28px;
    margin: 32px 0;
  }}

  .finding-label {{
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 10px;
  }}

  .finding p {{
    font-size: 13px;
    color: var(--text);
    line-height: 1.8;
  }}

  /* ── PLACEHOLDER ── */
  .placeholder {{
    background: var(--card);
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 60px;
    text-align: center;
    color: var(--muted);
    font-size: 12px;
  }}

  /* ── FOOTER ── */
  footer {{
    position: relative;
    z-index: 1;
    padding: 40px 60px;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
  }}

  footer p {{ font-size: 11px; color: var(--muted); }}
  footer a {{ color: var(--accent); text-decoration: none; }}
</style>
</head>
<body>

<!-- HERO -->
<section class="hero">
  <p class="hero-tag">Research Demo · JEPA-Lens</p>
  <h1>What Does a Model<br><span>Understand</span>?</h1>
  <p class="hero-sub">
    A from-scratch implementation of Image JEPA that maps model understanding
    via prediction error in representation space — no labels, no gradients.
    Four training variants reveal how steering the objective shifts what the model finds complex.
  </p>
  <div class="hero-stats">
    <div class="stat">
      <span class="stat-val">4</span>
      <span class="stat-label">Models trained</span>
    </div>
    <div class="stat">
      <span class="stat-val">51.4%</span>
      <span class="stat-label">Linear probe acc</span>
    </div>
    <div class="stat">
      <span class="stat-val">3.1M</span>
      <span class="stat-label">Parameters</span>
    </div>
    <div class="stat">
      <span class="stat-val">0</span>
      <span class="stat-label">Labels used</span>
    </div>
  </div>
</section>

<!-- NAV -->
<nav class="nav">
  <button class="tab active" onclick="switchTab('understanding')">Understanding Maps</button>
  <button class="tab" onclick="switchTab('compare')">Variant Comparison</button>
  <button class="tab" onclick="switchTab('masking')">Masking Viz</button>
  <button class="tab" onclick="switchTab('stats')">Training Stats</button>
</nav>

<!-- MAIN -->
<main class="main">

  <!-- ── TAB 1: UNDERSTANDING MAPS ── -->
  <div id="tab-understanding" class="section active">
    <div class="section-header">
      <h2 class="section-title">Understanding Maps</h2>
      <p class="section-desc">
        Per-patch prediction error heatmaps. Each patch is hidden one at a time —
        the error measures how hard it was to predict from context alone.
        Red = complex/uncertain. Green = predictable/trivial.
        Object regions emerge as red without any labels.
      </p>
    </div>

    <div class="finding">
      <p class="finding-label">Key insight</p>
      <p>The encoder never saw a label. Yet it found dog faces harder to predict than sky,
      car bodies harder than road. Semantic complexity emerges purely from the
      self-supervised prediction objective.</p>
    </div>

    <!-- Model selector -->
    <div class="model-grid">
      <div class="model-card selected" onclick="showUmap('baseline')" id="card-baseline">
        <div class="model-card-header">
          <span class="model-name">Baseline v2</span>
          <span class="model-acc">51.4%</span>
        </div>
        <p class="model-desc">Standard JEPA · depth 6 · 60 epochs · multi-crop masking</p>
      </div>
      <div class="model-card" onclick="showUmap('noise')" id="card-noise">
        <div class="model-card-header">
          <span class="model-name">Noise-Robust</span>
          <span class="model-acc">49.9%</span>
        </div>
        <p class="model-desc">Encoder sees noisy input · target sees clean · forces global reasoning</p>
      </div>
      <div class="model-card" onclick="showUmap('structure')" id="card-structure">
        <div class="model-card-header">
          <span class="model-name">Structure-Focused</span>
          <span class="model-acc">50.3%</span>
        </div>
        <p class="model-desc">Canny edge detection on input · shifts attention to shapes over texture</p>
      </div>
      <div class="model-card" onclick="showUmap('highmask')" id="card-highmask">
        <div class="model-card-header">
          <span class="model-name">High-Mask</span>
          <span class="model-acc">49.1%</span>
        </div>
        <p class="model-desc">75% patches hidden · encoder sees only 25% · learns global patterns</p>
      </div>
    </div>

    <!-- Image display -->
    {'<div class="img-container" id="umap-display"><div class="img-header"><span id="umap-title">BASELINE V2</span><span>red = complex · green = predictable</span></div><img id="umap-img" src="' + (images['umap_baseline'] or '') + '" alt="Understanding map"></div>' if images['umap_baseline'] else '<div class="placeholder">Run evaluate.py to generate understanding maps</div>'}
  </div>

  <!-- ── TAB 2: VARIANT COMPARISON ── -->
  <div id="tab-compare" class="section">
    <div class="section-header">
      <h2 class="section-title">Variant Comparison</h2>
      <p class="section-desc">
        Side by side: how does steering the training objective change what the model finds complex?
        The delta between maps — not the absolute error — is the research finding.
        Regions that are red in one variant but green in another reveal what each training condition genuinely learned.
      </p>
    </div>

    <div class="finding">
      <p class="finding-label">What to look for</p>
      <p>Sky stays green across all variants — trivially predictable regardless of training.
      Object regions shift between variants — those shifts reveal genuine steering of understanding.</p>
    </div>

    <div class="compare-grid">
      {'<div class="img-container"><div class="img-header"><span>Baseline v2</span><span>51.4%</span></div><img src="' + images['umap_baseline'] + '" alt="Baseline"></div>' if images['umap_baseline'] else '<div class="placeholder">baseline missing</div>'}
      {'<div class="img-container"><div class="img-header"><span>Noise-Robust</span><span>49.9%</span></div><img src="' + images['umap_noise'] + '" alt="Noise"></div>' if images['umap_noise'] else '<div class="placeholder">noise_robust missing</div>'}
      {'<div class="img-container"><div class="img-header"><span>Structure-Focused</span><span>50.3%</span></div><img src="' + images['umap_structure'] + '" alt="Structure"></div>' if images['umap_structure'] else '<div class="placeholder">structure_focused missing</div>'}
      {'<div class="img-container"><div class="img-header"><span>High-Mask</span><span>49.1%</span></div><img src="' + images['umap_highmask'] + '" alt="High mask"></div>' if images['umap_highmask'] else '<div class="placeholder">high_mask missing</div>'}
    </div>
  </div>

  <!-- ── TAB 3: MASKING VIZ ── -->
  <div id="tab-masking" class="section">
    <div class="section-header">
      <h2 class="section-title">Masking Visualization</h2>
      <p class="section-desc">
        What the encoder actually saw vs what it had to predict.
        Blue patches = context (visible to encoder).
        Red patches = target (hidden, must be predicted from context alone).
        Row 3 shows prediction error per hidden patch.
      </p>
    </div>

    {'<div class="img-container"><div class="img-header"><span>BASELINE V2</span><span>row 1: original · row 2: encoder input · row 3: prediction error</span></div><img src="' + images['masking'] + '" alt="Masking visualization"></div>' if images['masking'] else '<div class="placeholder">Run python visualize_masking.py --checkpoint checkpoints/baseline_v2.pth</div>'}
  </div>

  <!-- ── TAB 4: TRAINING STATS ── -->
  <div id="tab-stats" class="section">
    <div class="section-header">
      <h2 class="section-title">Training Stats</h2>
      <p class="section-desc">
        Linear probe accuracy across all 4 models. All variants within ~2% of each other —
        fine-tuning steers representations without destroying them.
        The accuracy number is a sanity check. The heatmaps are the finding.
      </p>
    </div>

    <div class="acc-bars">
      <div class="acc-row">
        <span class="acc-label">Baseline v2</span>
        <div class="acc-bar-wrap"><div class="acc-bar-fill" style="width:51.4%"></div></div>
        <span class="acc-val">51.4%</span>
      </div>
      <div class="acc-row">
        <span class="acc-label">Noise-Robust</span>
        <div class="acc-bar-wrap"><div class="acc-bar-fill" style="width:49.9%"></div></div>
        <span class="acc-val">49.9%</span>
      </div>
      <div class="acc-row">
        <span class="acc-label">Structure-Focused</span>
        <div class="acc-bar-wrap"><div class="acc-bar-fill" style="width:50.3%"></div></div>
        <span class="acc-val">50.3%</span>
      </div>
      <div class="acc-row">
        <span class="acc-label">High-Mask</span>
        <div class="acc-bar-wrap"><div class="acc-bar-fill" style="width:49.1%"></div></div>
        <span class="acc-val">49.1%</span>
      </div>
      <div class="acc-row">
        <span class="acc-label" style="color:#555">Random baseline</span>
        <div class="acc-bar-wrap"><div class="acc-bar-fill" style="width:10%;background:var(--muted)"></div></div>
        <span class="acc-val" style="color:var(--muted)">10.0%</span>
      </div>
    </div>

    <div class="finding">
      <p class="finding-label">Why accuracy is not the point</p>
      <p>All models score ~50% — 5× above random chance — using zero labels.
      The variants don't improve accuracy because accuracy measures classification,
      not understanding. What changes between variants is the structure of the
      representation space — visible in the heatmaps, not the numbers.</p>
    </div>

    <!-- Loss curves -->
    <h3 style="font-family:'Syne',sans-serif; font-size:16px; margin-bottom:20px; color:var(--muted); font-weight:600;">Loss Curves</h3>
    <div class="compare-grid">
      {'<div class="img-container"><div class="img-header"><span>Baseline v2</span><span>60 epochs</span></div><img src="' + images['loss_baseline'] + '" alt="Loss baseline"></div>' if images['loss_baseline'] else '<div class="placeholder">loss_curve_v2.png missing</div>'}
      {'<div class="img-container"><div class="img-header"><span>Noise-Robust</span><span>20 epochs fine-tune</span></div><img src="' + images['loss_noise'] + '" alt="Loss noise"></div>' if images['loss_noise'] else '<div class="placeholder">loss_curve_noise_robust.png missing</div>'}
      {'<div class="img-container"><div class="img-header"><span>Structure-Focused</span><span>20 epochs fine-tune</span></div><img src="' + images['loss_structure'] + '" alt="Loss structure"></div>' if images['loss_structure'] else '<div class="placeholder">loss_curve_structure_focused.png missing</div>'}
      {'<div class="img-container"><div class="img-header"><span>High-Mask</span><span>20 epochs fine-tune</span></div><img src="' + images['loss_highmask'] + '" alt="Loss highmask"></div>' if images['loss_highmask'] else '<div class="placeholder">loss_curve_high_mask.png missing</div>'}
    </div>
  </div>

</main>

<footer>
  <p>JEPA-Lens — from-scratch Image JEPA on CIFAR-10 · 3.1M params · zero labels</p>
  <p><a href="https://arxiv.org/abs/2301.08243" target="_blank">Assran et al. 2023</a> · <a href="https://github.com/mukundhr/JEPA-Lens" target="_blank">github.com/mukundhr/JEPA-Lens</a></p>
</footer>

<script>
  // Tab switching
  function switchTab(name) {{
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    event.target.classList.add('active');
  }}

  // Understanding map selector
  const umaps = {{
    baseline:  {{ src: "{images.get('umap_baseline', '')  or ''}", title: "BASELINE V2" }},
    noise:     {{ src: "{images.get('umap_noise', '')     or ''}", title: "NOISE-ROBUST" }},
    structure: {{ src: "{images.get('umap_structure', '') or ''}", title: "STRUCTURE-FOCUSED" }},
    highmask:  {{ src: "{images.get('umap_highmask', '')  or ''}", title: "HIGH-MASK" }},
  }};

  function showUmap(variant) {{
    const img   = document.getElementById('umap-img');
    const title = document.getElementById('umap-title');
    if (img && umaps[variant].src) {{
      img.src   = umaps[variant].src;
      title.textContent = umaps[variant].title;
    }}
    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
    document.getElementById('card-' + variant).classList.add('selected');
  }}
</script>
</body>
</html>"""

# ── WRITE FILE ────────────────────────────────────────────────────────────────
out_path = "jepa_lens_dashboard.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

size_mb = os.path.getsize(out_path) / (1024 * 1024)
print(f"✓ Dashboard saved → {out_path}  ({size_mb:.1f} MB)")
print(f"  Open in any browser — no server needed.")
print(f"\n  Missing images: {missing if missing else 'none — all good!'}")