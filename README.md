
# Phishing Email Detection (RoBERTa Email Classifier)

Team Repo for ICT30016 Innovation Concept.

A deep learning project that uses RoBERTa to classify emails as either phishing or legitimate.
This repository is designed for collaboration and collation of the ML/DL model.

## Setup
```bash
# 1) Clone the Repo

git clone https://github.com/twigvs/AI-Phishing-Email-Detection.git
cd AI-Phishing-Email-Detection

# 2) Create the Virtual Environment
Please use Python 3.12, unfortunately 3.13 does not yet support PyTorch

python3.12 -m venv venv
source venv/bin/activate    (if on MacOS or Linux)
venv\Scripts\activate       (if on Windows)

# 3) Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt
```

## Repo layout
```
phishguard/
AI-Phishing-Email-Detection/
├─ api/
│  └─ app.py                # Flask API (predict + evaluate)
├─ models/                  # (optional) local model folders (HF or sklearn)
│  ├─ roberta/              # e.g., saved HF model (config.json, tokenizer.json, model.safetensors)
│  ├─ model.pkl             # sklearn classifier (optional)
│  └─ vectorizer.pkl        # sklearn vectorizer (optional)
├─ pygui.py                 # GUI with verdict badge and gauge
├─ requirements.txt         # (optional) pinned deps
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
This project is for assessment within Swinburne University of Technology, Australia.
