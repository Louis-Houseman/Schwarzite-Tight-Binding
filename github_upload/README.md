# Schwarzite tight-binding (upload bundle)

This folder is **self-contained**: push its contents as the **root** of a new GitHub repo (or upload the whole folder).

## Contents

| Item | Purpose |
|------|---------|
| `schwarzite_render.py` | Main CLI: TB models, true first-BZ folding for `H(k)`, bands, optional topology tasks |
| `data/cif/AFY_nozeolite_relaxed.cif` | Primitive cell geometry (required) |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Ignores `results/`, bytecode |

## Quick start

```bash
cd github_upload   # or whatever you named the repo root after clone
python -m venv .venv
```

Windows: `.venv\Scripts\activate`  
macOS/Linux: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
python schwarzite_render.py
```

Run from **this folder** so generated `figures/` and `results/` land here. The CIF path is resolved relative to `schwarzite_render.py`, so the structure file is found as long as `data/cif/` stays alongside the script.

## GitHub

From this directory:

```bash
git init
git add schwarzite_render.py requirements.txt README.md .gitignore data/
git commit -m "Initial schwarzite TB exploration bundle"
```

Then create an empty repo on GitHub and `git remote add origin …` / `git push`.
