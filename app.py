"""SVD Compression Lab: a local app comparing plain vs. activation-whitened SVD on GPT-2.

Run:  python app.py      then open http://127.0.0.1:7860 (it opens automatically)

First launch downloads GPT-2 small + WikiText-2 and calibrates (a few minutes on CPU, under a
minute on a GPU). Calibration statistics are cached in ./cache so later launches start faster.
"""
import argparse
import os
from collections import OrderedDict

import gradio as gr
import numpy as np
import torch
from matplotlib.figure import Figure

import svd_core as core

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_LEN = 1024
PLAIN_C, WHITE_C, BASE_C = "#d1495b", "#2e86ab", "#7a7a7a"
LABELS = {"plain": "Plain SVD (keep W close)", "whitened": "Whitened SVD (keep WX close)"}


# ====================================================================== model state
class Lab:
    def __init__(self, model, tok, train_ids, test_ids, n_calib, calib_len, eval_tokens,
                 cache_dir=None, progress=print):
        self.base, self.tok = model.eval(), tok
        self.test_ids = test_ids[:eval_tokens]
        grams = self._load_or_collect_grams(train_ids, n_calib, calib_len, cache_dir, progress)
        progress("Computing SVDs of W and WL for every weight matrix ...")
        self.svds = core.build_layer_svds(model, grams, DEVICE,
                                          progress=lambda f, d: progress(f"  [{f:4.0%}] {d}"))
        self.names = list(self.svds)
        self.by_label = {core.short_name(n): n for n in self.names}
        self.base_params = core.matrix_params(model)
        self._models = OrderedDict()
        self._ppl = {}

    def _load_or_collect_grams(self, train_ids, n_calib, calib_len, cache_dir, progress):
        path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            path = os.path.join(cache_dir, f"grams_gpt2_{n_calib}x{calib_len}.pt")
            if os.path.exists(path):
                progress(f"Loading cached calibration statistics from {path}")
                return {k: v.double() for k, v in torch.load(path).items()}
        progress(f"Calibrating on {n_calib} WikiText-2 windows of {calib_len} tokens ...")
        grams = core.collect_grams(self.base, train_ids, n_calib, calib_len, DEVICE)
        if path:
            torch.save({k: v.float().cpu() for k, v in grams.items()}, path)
            progress(f"Saved calibration cache to {path}")
        return grams

    def model(self, ratio, method):
        if ratio <= 0:
            return self.base
        key = (round(ratio, 3), method)
        if key in self._models:
            self._models.move_to_end(key)
            return self._models[key]
        m = core.compress(self.base, self.svds, ratio, method)
        self._models[key] = m
        while len(self._models) > 4:          # keep a few compressed copies in memory
            self._models.popitem(last=False)
        return m

    def ppl(self, ratio, method):
        key = ("original",) if ratio <= 0 else (round(ratio, 3), method)
        if key not in self._ppl:
            self._ppl[key] = core.perplexity(self.model(ratio, method), self.test_ids, EVAL_LEN, DEVICE)
        return self._ppl[key]

    @torch.no_grad()
    def generate(self, model, prompt, n_tokens, seed):
        torch.manual_seed(seed)
        enc = self.tok(prompt, return_tensors="pt")
        out = model.generate(enc.input_ids.to(DEVICE), attention_mask=enc.attention_mask.to(DEVICE),
                             max_new_tokens=n_tokens, do_sample=True, top_k=50,
                             pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0], skip_special_tokens=True)


# ====================================================================== plotting helpers
def _fig(w=7, h=4):
    fig = Figure(figsize=(w, h), dpi=110)
    ax = fig.subplots()
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def ranks_note(lab, ratio):
    shapes = sorted({s.shape for s in lab.svds.values()})
    parts = [f"{m}×{n} → k = {core.rank_for_ratio(m, n, ratio)} (break-even {m * n // (m + n)})"
             for m, n in shapes]
    return " · ".join(parts)


# ====================================================================== tab handlers
def make_handlers(lab):
    def compare(pct, prompt, n_tokens, seed, measure, progress=gr.Progress()):
        ratio = pct / 100
        rows, texts = [], []
        runs = [("Original", 0.0, "plain"), (LABELS["plain"], ratio, "plain"),
                (LABELS["whitened"], ratio, "whitened")]
        for i, (label, r, method) in enumerate(runs):
            progress(i / len(runs), desc=f"{label}: compressing / generating")
            model = lab.model(r, method)
            texts.append(lab.generate(model, prompt, int(n_tokens), int(seed)))
            params = core.matrix_params(model)
            ppl = f"{lab.ppl(r, method):,.2f}" if measure else "—"
            rows.append(f"| {label} | {params:,} | {params / lab.base_params:.0%} | {ppl} |")
        table = ("| Model | Block weight params | Size | Perplexity ↓ |\n|---|---:|---:|---:|\n"
                 + "\n".join(rows)
                 + f"\n\n**Ranks at {pct:.0f}% removed:** {ranks_note(lab, ratio)}"
                 + (f"\n\nPerplexity on the first {len(lab.test_ids):,} WikiText-2 test tokens."
                    if measure else ""))
        return table, *texts

    def sweep(pcts, progress=gr.Progress()):
        pcts = sorted(int(p) for p in pcts)
        if not pcts:
            raise gr.Error("Pick at least one compression level.")
        base = lab.ppl(0, "plain")
        res = {m: [] for m in core.METHODS}
        total = len(pcts) * 2
        for i, p in enumerate(pcts):
            for j, m in enumerate(core.METHODS):
                progress((2 * i + j) / total, desc=f"{p}% removed · {m}")
                res[m].append(lab.ppl(p / 100, m))
        fig, ax = _fig()
        ax.plot(pcts, res["plain"], "o-", c=PLAIN_C, label=LABELS["plain"])
        ax.plot(pcts, res["whitened"], "s-", c=WHITE_C, label=LABELS["whitened"])
        ax.axhline(base, ls="--", c=BASE_C, label=f"Uncompressed ({base:.1f})")
        ax.set_yscale("log")
        ax.set_xlabel("Parameters removed from each weight matrix (%)")
        ax.set_ylabel("Perplexity (log scale, lower is better)")
        ax.set_title("GPT-2 small: same rank, different target")
        ax.legend()
        fig.tight_layout()
        table = ("| Removed | Plain SVD | Whitened SVD |\n|---:|---:|---:|\n"
                 + "\n".join(f"| {p}% | {a:,.2f} | {b:,.2f} |"
                             for p, a, b in zip(pcts, res["plain"], res["whitened"])))
        return fig, table

    def layer_view(label, pct):
        ratio = pct / 100
        name = lab.by_label[label]
        s = lab.svds[name]
        m, n = s.shape
        k = core.rank_for_ratio(m, n, ratio)

        fig1, ax = _fig()
        for meth, c in (("plain", PLAIN_C), ("whitened", WHITE_C)):
            sv = s.sv[meth].cpu().numpy()
            ax.semilogy(np.arange(1, len(sv) + 1), sv / sv[0], c=c,
                        label=f"σᵢ({'W' if meth == 'plain' else 'WL'})")
        ax.axvline(k, ls=":", c="k", label=f"kept rank k = {k}")
        ax.axvline(m * n / (m + n), ls="--", c=BASE_C, label=f"break-even {m * n // (m + n)}")
        ax.set_xlabel("index i")
        ax.set_ylabel("σᵢ / σ₁ (log scale)")
        ax.set_title(f"Singular values · {label} ({m}×{n})")
        ax.legend()
        fig1.tight_layout()

        rows = []
        for meth in core.METHODS:
            w_err, o_err = s.errors(meth, k)
            rows.append(f"| {LABELS[meth]} | {w_err:.4f} | {o_err:.4f} |")
        table = (f"**{label}**, k = {k} of {min(m, n)}\n\n"
                 "| Method | Weight error ‖W−W′‖/‖W‖ | Output error ‖(W−W′)X‖/‖WX‖ |\n|---|---:|---:|\n"
                 + "\n".join(rows))

        errs = {meth: [] for meth in core.METHODS}
        for nm in lab.names:
            sm, sn = lab.svds[nm].shape
            kk = core.rank_for_ratio(sm, sn, ratio)
            for meth in core.METHODS:
                errs[meth].append(lab.svds[nm].errors(meth, kk)[1])
        x = np.arange(len(lab.names))
        fig2 = Figure(figsize=(12, 4.2), dpi=110)
        ax2 = fig2.subplots()
        ax2.bar(x - 0.2, errs["plain"], 0.4, color=PLAIN_C, label=LABELS["plain"])
        ax2.bar(x + 0.2, errs["whitened"], 0.4, color=WHITE_C, label=LABELS["whitened"])
        ax2.set_xticks(x, [core.short_name(nm).replace("block ", "") for nm in lab.names],
                       rotation=90, fontsize=7)
        ax2.set_yscale("log")
        ax2.set_ylabel("Relative output error (log)")
        wins = sum(w < p for w, p in zip(errs["whitened"], errs["plain"]))
        ax2.set_title(f"Output error for every matrix at {pct:.0f}% removed "
                      f"(whitened lower on {wins}/{len(x)})")
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.legend()
        fig2.tight_layout()
        return fig1, table, fig2

    return compare, sweep, layer_view


# ====================================================================== image tab
def image_factor(img):
    if img is None:
        return None
    a = np.asarray(img, dtype=float)
    if a.ndim == 2:
        a = a[..., None]
    a = a[..., :3]
    if a.max() > 1:
        a = a / 255.0
    return [np.linalg.svd(a[..., c], full_matrices=False) for c in range(a.shape[-1])]


def image_rebuild(svds, k):
    if not svds:
        return None, "Upload an image.", None
    k = int(k)
    out = np.stack([(U[:, :k] * S[:k]) @ Vt[:k] for U, S, Vt in svds], axis=-1)
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    if out.shape[-1] == 1:
        out = out[..., 0]
    h, w = svds[0][0].shape[0], svds[0][2].shape[1]
    S_all = np.concatenate([S for _, S, _ in svds])
    energy = np.cumsum(np.sort(svds[0][1])[::-1] ** 2) / np.sum(svds[0][1] ** 2)
    kept = sum(np.sum(S[:k] ** 2) for _, S, _ in svds) / np.sum(S_all ** 2)
    storage = k * (h + w + 1) / (h * w)
    md = (f"**k = {k}** of {min(h, w)} · storage **{storage:.1%}** of the original "
          f"({h}×{w} per channel) · Frobenius energy kept **{kept:.2%}**  \n"
          f"Relative error ‖A−A_k‖/‖A‖ = √(dropped σ² / all σ²) = **{np.sqrt(max(0, 1 - kept)):.3f}**")
    fig, ax = _fig(6, 3.4)
    ax.plot(np.arange(1, len(energy) + 1), energy, c=WHITE_C)
    ax.scatter([k], [energy[min(k, len(energy)) - 1]], c=PLAIN_C, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("k (log scale)")
    ax.set_ylabel("Energy kept (first channel)")
    ax.set_title("Most of an image lives in a few singular values")
    fig.tight_layout()
    return out, md, fig


# ====================================================================== UI
INTRO = """
# SVD Compression Lab
Every weight matrix in GPT-2 small is replaced by a rank-k product **W′ = AB**, storing k(m+n) numbers instead of mn.
Two ways to pick A and B, same rank, same size:

- **Plain SVD** keeps the *weights* close: W′ = U_k Σ_k V_kᵀ minimizes ‖W − W′‖ (Eckart-Young on W).
- **Whitened SVD** (SVD-LLM style) keeps the *outputs on real data* close: with XXᵀ = LLᵀ from calibration text,
  W′ = U_k Σ_k V_kᵀ L⁻¹ from the SVD of WL minimizes ‖WX − W′X‖ (Eckart-Young on WL).

No fine-tuning anywhere, so the only difference is *what the decomposition keeps close*.
"""


def build_ui(lab, default_image=None):
    compare, sweep, layer_view = make_handlers(lab)
    labels = list(lab.by_label)

    with gr.Blocks(title="SVD Compression Lab") as demo:
        gr.Markdown(INTRO)

        with gr.Tab("Compare outputs"):
            with gr.Row():
                pct = gr.Slider(0, 60, value=30, step=5, label="% of each weight matrix removed")
                n_tokens = gr.Slider(10, 120, value=40, step=10, label="Tokens to generate")
                seed = gr.Number(value=0, precision=0, label="Sampling seed")
            prompt = gr.Textbox(value="The singular value decomposition is", label="Prompt")
            measure = gr.Checkbox(value=True, label="Also measure perplexity (slower, cached after first run)")
            run = gr.Button("Compress and generate", variant="primary")
            stats = gr.Markdown()
            with gr.Row():
                t_orig = gr.Textbox(label="Original GPT-2", lines=8)
                t_plain = gr.Textbox(label=LABELS["plain"], lines=8)
                t_white = gr.Textbox(label=LABELS["whitened"], lines=8)
            run.click(compare, [pct, prompt, n_tokens, seed, measure], [stats, t_orig, t_plain, t_white])

        with gr.Tab("Perplexity sweep"):
            levels = gr.CheckboxGroup([str(p) for p in (10, 20, 30, 40, 50, 60)],
                                      value=["10", "20", "30", "40"], label="% removed to test")
            go = gr.Button("Run sweep", variant="primary")
            sweep_plot = gr.Plot(label="Perplexity vs. compression")
            sweep_table = gr.Markdown()
            go.click(sweep, [levels], [sweep_plot, sweep_table])

        with gr.Tab("Inside a layer"):
            with gr.Row():
                layer = gr.Dropdown(labels, value=labels[len(labels) // 2], label="Weight matrix")
                lpct = gr.Slider(5, 60, value=20, step=5, label="% removed")
            spec = gr.Plot(label="Singular value spectrum")
            err_table = gr.Markdown()
            all_layers = gr.Plot(label="All layers")
            for ev in (layer.change, lpct.release):
                ev(layer_view, [layer, lpct], [spec, err_table, all_layers])
            demo.load(layer_view, [layer, lpct], [spec, err_table, all_layers])

        with gr.Tab("Image SVD"):
            gr.Markdown("Each color channel is a matrix. Keep its k largest singular values and rebuild. "
                        "Upload your own photo to try it.")
            state = gr.State(image_factor(default_image))
            with gr.Row():
                img_in = gr.Image(value=default_image, type="numpy", label="Original")
                img_out = gr.Image(type="numpy", label="Rank-k reconstruction")
            k_slider = gr.Slider(1, 200, value=20, step=1, label="k (singular values kept per channel)")
            img_md = gr.Markdown()
            energy_plot = gr.Plot(label="Energy curve")
            outs = [img_out, img_md, energy_plot]
            img_in.upload(image_factor, img_in, state).then(image_rebuild, [state, k_slider], outs)
            k_slider.change(image_rebuild, [state, k_slider], outs)
            demo.load(image_rebuild, [state, k_slider], outs)

    return demo


# ====================================================================== entry point
def main():
    ap = argparse.ArgumentParser(description="SVD Compression Lab (local)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--calib", type=int, default=64, help="calibration windows (more = better XXᵀ estimate)")
    ap.add_argument("--calib-len", type=int, default=512)
    ap.add_argument("--eval-tokens", type=int, default=16384,
                    help="WikiText-2 test tokens used for perplexity (full set is ~287k; slower)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print(f"Device: {DEVICE}")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_ids = tok("\n\n".join(wiki["train"]["text"]), return_tensors="pt").input_ids[0]
    test_ids = tok("\n\n".join(wiki["test"]["text"]), return_tensors="pt").input_ids[0]

    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    lab = Lab(model, tok, train_ids, test_ids, args.calib, args.calib_len, args.eval_tokens, cache)

    try:
        from skimage import data
        default_image = data.coffee()
    except Exception:
        default_image = None

    print("Ready.")
    build_ui(lab, default_image).queue().launch(server_name="127.0.0.1", server_port=args.port,
                                                inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
