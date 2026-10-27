# SVD Compression Lab

A local app for comparing two ways of compressing GPT-2 small with the singular value decomposition, at the same rank:

| Method | Factorization | What it keeps close |
|---|---|---|
| Plain SVD | W′ = U_k Σ_k V_kᵀ | the weights, ‖W − W′‖ |
| Whitened SVD (SVD-LLM style) | XXᵀ = LLᵀ, WL = UΣVᵀ, W′ = U_k Σ_k V_kᵀ L⁻¹ | the layer's outputs on real text, ‖WX − W′X‖ |

Neither method is fine-tuned, so any difference comes only from what the decomposition targets.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The app opens at http://127.0.0.1:7860 and runs entirely on your machine.

The first launch downloads GPT-2 small (~500 MB) and WikiText-2, then calibrates. That takes a few minutes on CPU and under a minute on an NVIDIA GPU. Calibration statistics are saved to `./cache`, so later launches skip that step.

For a CUDA GPU, install the CUDA build of PyTorch first (see pytorch.org for the command), then `pip install -r requirements.txt`.

### Options

```bash
python app.py --port 8000          # different port
python app.py --calib 128          # more calibration text (better XXᵀ estimate, slower first launch)
python app.py --eval-tokens 65536  # more test text for perplexity (slower, more precise)
python app.py --no-browser         # don't open a browser tab automatically
```

## Tabs

- **Compare outputs**: pick a compression level and a prompt, then see text from the original model and both compressed versions, with parameter counts and perplexity.
- **Perplexity sweep**: perplexity vs. % removed for both methods on one chart.
- **Inside a layer**: the singular value spectrum of W vs. WL for any of the 48 weight matrices, weight error vs. output error for each method, and output error across every layer.
- **Image SVD**: rebuild any photo from its top k singular values, with storage cost and Frobenius energy kept.

## Files

- `svd_core.py`: the math (calibration, Cholesky whitening, per-layer SVDs, low-rank layers, perplexity)
- `app.py`: the Gradio interface and command-line entry point

## Background

Companion to "Matrix Decomposition in Artificial Intelligence" (DSCI 369, Colorado State University), which compares Ben Noach & Goldberg (2020), Wang et al. (SVD-LLM, 2024), and Lin et al. (MoDeGPT, 2024).
