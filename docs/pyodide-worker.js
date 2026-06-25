/*
 * pyodide-worker.js
 *
 * Runs in a Web Worker so that Python execution never blocks the main
 * thread / UI. Handles three jobs, triggered by postMessage from the
 * main page:
 *   - init: load Pyodide + scikit-learn/numpy/pandas (one-time)
 *   - fetchRepo: pull a student's files from GitHub and write them into
 *     Pyodide's virtual filesystem, preserving folder structure
 *   - runMain / runTests: execute code, streaming stdout back
 */

importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js");

let pyodide = null;
let pyodideReadyPromise = null;

// The exact relative file paths we expect in a student's repo.
// Centralized here so the fetch step and the FS-writing step always agree.
const PROJECT_FILES = [
  "iris_classifier/__init__.py",
  "iris_classifier/data.py",
  "iris_classifier/model.py",
  "iris_classifier/evaluate.py",
  "iris_classifier/predict.py",
  "tests/__init__.py",
  "tests/test_data.py",
  "tests/test_model.py",
  "tests/test_evaluate.py",
  "tests/test_predict.py",
  "main.py",
];

function post(type, payload) {
  self.postMessage({ type, ...payload });
}

// Extract a readable string from any thrown value, including non-standard
// error shapes (e.g. Emscripten's FS.ErrnoError, or Pyodide's PythonError)
// that don't always stringify cleanly via String(err) or err.message.
function describeError(err) {
  if (!err) return "Unknown error";
  if (typeof err === "string") return err;
  if (err.message && typeof err.message === "string" && err.message.length > 0) {
    return err.message;
  }
  if (typeof err.toString === "function") {
    const s = err.toString();
    if (s && s !== "[object Object]") return s;
  }
  try {
    return JSON.stringify(err);
  } catch (e) {
    return "An error occurred that could not be described (" + Object.prototype.toString.call(err) + ")";
  }
}

async function initPyodide() {
  if (pyodideReadyPromise) return pyodideReadyPromise;

  pyodideReadyPromise = (async () => {
    post("status", { message: "Downloading Python runtime…" });
    pyodide = await loadPyodide();

    post("status", { message: "Installing scikit-learn, numpy, pandas…" });
    await pyodide.loadPackage(["numpy", "scikit-learn", "pandas"]);

    post("status", { message: "Installing pytest…" });
    await pyodide.loadPackage("micropip");
    const micropip = pyodide.pyimport("micropip");
    await micropip.install("pytest");

    post("ready", {});
  })();

  return pyodideReadyPromise;
}

// Fetch a single file's raw text from a public GitHub repo.
async function fetchRawFile(owner, repo, branch, path) {
  const url = `https://raw.githubusercontent.com/${owner}/${repo}/${branch}/${path}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Could not fetch ${path} (HTTP ${res.status}). Check the repo, branch, and that it's public.`);
  }
  return await res.text();
}

async function fetchRepoAndPopulate(owner, repo, branch) {
  await initPyodide();

  const files = {};
  let completed = 0;

  for (const relPath of PROJECT_FILES) {
    post("fetchProgress", { path: relPath, completed, total: PROJECT_FILES.length });
    const content = await fetchRawFile(owner, repo, branch, relPath);
    files[relPath] = content;
    completed += 1;
  }

  // Write the fresh set of files into the virtual filesystem, creating
  // any needed subdirectories first via mkdirTree (safe to call even if
  // the directory already exists from a previous load).
  for (const relPath of PROJECT_FILES) {
    const dir = relPath.includes("/") ? relPath.split("/").slice(0, -1).join("/") : null;
    if (dir) {
      try {
        pyodide.FS.mkdirTree(dir);
      } catch (e) {
        // directory already exists from a previous "Load repo" click - fine
      }
    }
    try {
      pyodide.FS.writeFile(relPath, files[relPath]);
    } catch (e) {
      throw new Error(`Failed writing ${relPath} into the Python filesystem: ${describeError(e)}`);
    }
  }

  post("repoLoaded", { files });
}

async function runMain() {
  await initPyodide();

  let output = "";
  pyodide.setStdout({ batched: (msg) => { output += msg + "\n"; } });
  pyodide.setStderr({ batched: (msg) => { output += msg + "\n"; } });

  try {
    // Modules from a previous run may still be cached in sys.modules. If
    // the student re-loads an updated repo, stale cached code would
    // otherwise silently run instead of the freshly-fetched files.
    await pyodide.runPythonAsync(`
import sys
if "." not in sys.path:
    sys.path.insert(0, ".")
for mod_name in list(sys.modules):
    if mod_name == "iris_classifier" or mod_name.startswith("iris_classifier."):
        del sys.modules[mod_name]
`);
    const code = pyodide.FS.readFile("main.py", { encoding: "utf8" });
    // main.py is written with `if __name__ == "__main__": run()`. When run
    // via runPythonAsync, the module-level __name__ is not "__main__", so
    // that guard never fires. Set __name__ explicitly before exec'ing so
    // the script behaves exactly as it would from a real command line.
    pyodide.globals.set("__name__", "__main__");
    await pyodide.runPythonAsync(code);
    post("runComplete", { kind: "main", output, success: true });
  } catch (err) {
    post("runComplete", { kind: "main", output: output + "\n" + describeError(err), success: false });
  } finally {
    pyodide.setStdout({});
    pyodide.setStderr({});
  }
}

async function runTests() {
  await initPyodide();

  let output = "";
  pyodide.setStdout({ batched: (msg) => { output += msg + "\n"; } });
  pyodide.setStderr({ batched: (msg) => { output += msg + "\n"; } });

  try {
    await pyodide.runPythonAsync(`
import sys
if "." not in sys.path:
    sys.path.insert(0, ".")
for mod_name in list(sys.modules):
    if mod_name == "iris_classifier" or mod_name.startswith("iris_classifier."):
        del sys.modules[mod_name]
import pytest
exit_code = int(pytest.main(["tests", "-v", "--no-header", "-p", "no:cacheprovider"]))
`);
    const exitCode = pyodide.globals.get("exit_code");
    post("runComplete", { kind: "tests", output, success: exitCode === 0 });
  } catch (err) {
    post("runComplete", { kind: "tests", output: output + "\n" + describeError(err), success: false });
  } finally {
    pyodide.setStdout({});
    pyodide.setStderr({});
  }
}

self.onmessage = async (event) => {
  const { type, payload } = event.data;

  try {
    if (type === "init") {
      await initPyodide();
    } else if (type === "fetchRepo") {
      await fetchRepoAndPopulate(payload.owner, payload.repo, payload.branch);
    } else if (type === "runMain") {
      await runMain();
    } else if (type === "runTests") {
      await runTests();
    }
  } catch (err) {
    post("error", { message: describeError(err) });
  }
};
