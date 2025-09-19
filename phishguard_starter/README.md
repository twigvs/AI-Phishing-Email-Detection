
# PhishGuard (RoBERTa Email Classifier)

Team repo for ICT30016 Innovation Concept.

## Quickstart
```bash
# 1) Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 3) Run the baseline or RoBERTa training (see src/train.py)
# (You will need a dataset CSV with 'text' and 'label' columns.)
```

## Repo layout
```
phishguard/
├─ data/
│  ├─ raw/         # keep original data here (do NOT commit large files)
│  └─ processed/   # cleaned/split data (small samples only)
├─ notebooks/      # EDA and experiments (pair with Jupytext if possible)
├─ src/            # python modules (data loading, model, train, eval)
├─ configs/        # YAML configs (hyperparams etc.)
├─ tests/          # unit tests
├─ out/            # training outputs/checkpoints (gitignored)
└─ README.md
```

## Contributing
- Create a feature branch: `git checkout -b feat/your-task`
- Commit small changes with clear messages.
- Push branch and open a Pull Request (PR) on GitHub.
- Request at least one review.
- Use **Squash & merge** when approved; delete the branch after.

## Jupyter notebooks
To avoid merge pain, either:
- Use **Jupytext** to keep a `.py` twin of each notebook, or
- Ensure each person edits different notebooks.

## License
MIT (or update as your team prefers).
