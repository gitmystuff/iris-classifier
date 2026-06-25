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

## Running and testing your code in the browser (no install required)

Once your fork exists, you can run `main.py` and your `pytest` suite right in
the browser — no Python install needed — using a small tool built for this
project. This is a separate page from vscode.dev; vscode.dev is for *editing*,
this page is for *running*.

### Step 1: Turn on the page for your fork

Every fork gets its own copy of this tool, but it has to be switched on once:

1. On your forked repo's GitHub page, click **Settings**
2. In the left sidebar, click **Pages**
3. Under **Build and deployment → Source**, choose **Deploy from a branch**
4. Set **Branch** to `main` and the folder to **`/docs`**
5. Click **Save**

GitHub will show a green box with your page's address after about a minute.
It will look like:

```
https://<your-username>.github.io/<your-fork-name>/
```

### Step 2: Point the page at your own fork

The page is pre-filled to look at the original `gitmystuff/iris-classifier`
repo, not yours. There are two ways to fix that — pick whichever you're more
comfortable with.

**Option A — just retype it (easiest, no editing required)**
Every time you open the page, replace the text in the **first box** (next to
the GitHub logo icon) with your own GitHub username, then click **Load repo**.
You'll need to do this each time you open the page.

**Option B — edit one line so it remembers your username (recommended)**
This makes the page open *already* pointed at your fork, every time.

1. In **vscode.dev**, open your fork and find the file `docs/index.html`
2. Use `Ctrl+F` (or `Cmd+F` on a Mac) to search the file for: `gitmystuff`
3. You'll land on a line that looks like this:

   ```html
   <input type="text" name="owner" placeholder="github-username" value="gitmystuff" required />
   ```

4. Carefully replace **only the text between the quotes after `value=`**
   (currently `gitmystuff`) with your own GitHub username. For example, if
   your GitHub username is `jsmith42`, the line should become:

   ```html
   <input type="text" name="owner" placeholder="github-username" value="jsmith42" required />
   ```

   Don't change anything else on that line — just the username inside the
   quotes.
5. Save the file (`Ctrl+S` / `Cmd+S`), then commit and push using the
   **Source Control** panel, the same way you would for any other change.

If something looks broken after saving, undo your change (`Ctrl+Z`) and try
again — it's just a text file, nothing is destroyed by trying again.

### Step 3: Run your code

Open your page's address from Step 1. You should see:

- A green status dot and the message **"Python runtime ready"** after a few
  seconds (the first load is the slowest; it's downloading a Python
  environment into your browser)
- Click **Load repo** — your files will be fetched from GitHub and listed on
  the left
- Click **▶ Run main.py** to see your classifier's output
- Click **▶ Run pytest** to see whether your tests pass

If you see a red error message instead, read it — it usually names the exact
file or problem. Common first-time issues:

- **A fetch/HTTP error** — double-check your username is spelled correctly
  and your fork is set to **public** (Settings → General → Danger Zone →
  "Change visibility" if it isn't)
- **Pytest or main.py errors** — these are real bugs in your code, the same
  as you'd see running `pytest` locally; read the traceback the same way

## Note

* Vibe Coding -> Context Engineering -> Spec Driven Development -> Event Driven Code Agent
* Have the AI write the test first, the spec, the context, and then the code
