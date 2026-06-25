# Iris Classifier — From Notebook to Modules

This repo demonstrates two ways of writing the same data science project:

1. **`exploration.ipynb`** — the notebook / Colab-style approach: data loading, EDA,
   model building, and evaluation all mixed together in one iterative, exploratory
   document. Good for figuring things out.
2. **`iris_classifier/` + `tests/`** — the same logic, refactored into separated,
   independently-testable modules. Good for code that needs to be trusted, reused,
   and maintained.

Compare the two. Nothing in the notebook is "wrong" — it's just a different phase
of the work, with different goals.

## Project structure

```
iris-classifier/
├── exploration.ipynb         # Phase 1: notebook-style exploration
├── main.py                   # Phase 2: orchestrates the modules below
├── iris_classifier/
│   ├── data.py               # loading + splitting the dataset
│   ├── model.py               # building + training the classifier
│   ├── evaluate.py           # scoring predictions
│   └── predict.py            # predicting on one new sample
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_evaluate.py
│   └── test_predict.py
└── requirements.txt
```

## Running it locally

```bash
pip install -r requirements.txt
python main.py          # runs the full pipeline
pytest tests/ -v        # runs the test suite
```

## Editing this in the browser (no install required)

You can browse, fork, and edit this entire project in [vscode.dev](https://vscode.dev)
without installing anything locally:

1. **Browse the repo in vscode.dev**
   Open: `https://vscode.dev/github/<owner>/<repo>`
   (replace `<owner>/<repo>` with this repository's path)

2. **Fork it to your own GitHub account**
   - Click **Fork** on this repo's GitHub page, *or*
   - Use the **Source Control** panel inside vscode.dev to publish your changes
     to a new repository under your own account.

3. **Open your fork in vscode.dev**
   Open: `https://vscode.dev/github/<your-username>/<your-fork-name>`

4. **Edit the code**
   Make your changes directly in the browser editor. Full syntax highlighting
   and IntelliSense work, just like desktop VS Code.

5. **Commit and push**
   Use the **Source Control** panel (the icon in the left sidebar that looks
   like a branch) to stage, commit, and push your changes — all without a
   terminal or local git install.

6. **Submit your fork's URL** for evaluation.

## Your assignment

Starting from this **working** codebase, extend or refactor it. For example:

- Add a new module or function (e.g. try a different classifier in `model.py`)
- Add new test cases that cover an edge case the current tests miss
- Improve `predict.py` to validate its inputs (e.g. reject negative measurements)

Make sure `pytest tests/ -v` still passes (and ideally has *more* passing tests
than it started with) before you submit.
