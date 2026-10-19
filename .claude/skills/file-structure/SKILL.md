---
name: file-structure
description: Check and refine where PG_005 code lives -- map who calls what across the repo, list placement violations (shared code inside one pipeline, functions/ importing classes/, scripts importing another script's local functions, methods outside their class, split jobs), propose moves, apply only after approval, verify every affected pipeline. Use when the user says "check the structure", "file structure", "where should this live", "shared functions", "decouple", or before a big refactor.
---

## What this skill does

Finds code that sits in the wrong file and moves it, without changing behavior. Works across files (layout inside one file is `neat-refactor`).

1. Show that this skill is triggered.
2. Scope: ask which pipeline(s) / files, or propose one.
3. **Map callers.** For every module-level function, class and constant in scope, grep who imports or calls it (exclude `docs/`, `archive/`). Note the paper scripts separately: they are not updated by refactors.
4. **List violations** of the rules below as a table: `name | lives in | used by | rule | proposed home`. Give a real example for any new class (what it holds, one call).
5. **Wait for approval.** Approving the table is not approval of a design that isn't in it.
6. Apply one approved move at a time: move the code, update every importer, delete the old copy.
7. `ruff check` every touched file (not `uv run ruff`).
8. Verify (see "Verification"), report, state full output paths.

## Rules -- who calls it decides where it lives

| # | Situation | Home |
|---|---|---|
| 1 | Used by more than one pipeline / file | `functions/`; every caller imports it from there |
| 2 | Shared functions that form one object (own data + steps on it) | a class in `classes/<name>.py`, **preferred over loose functions**; methods named by step (`build_tree`, `best_silhouette`, `clustering`) |
| 3 | Used by one class only | a method next to its caller (`@staticmethod` if it reads no `self`) |
| 4 | Takes the class instance as argument (`fn(analyzer)`) | a method of that class |
| 5 | Used by one script only | that script, under its stage banner; no new `functions/` module |
| 6 | numba `@njit` / `@cuda.jit` kernels | module level in `functions/`, CPU + GPU (numba can't compile methods) |
| 7 | Constant shared by pipelines | next to the shared function that uses it (`CROSSOVER_RATIO` in `functions/fit_hist.py`) |
| 8 | Saving code for data that is reused later | a method of the class that owns the data (`save_flash_footprints`) |

Coupling that is always a violation:
- `functions/` importing from `classes/`.
- A script importing another script's local function (e.g. `from SRE_analysis import select_recordings`). Usually the job is already covered by a shared function (`filelist_parser`); make the shared part a function / class in `functions/` or `classes/` and let every script use it.

Shape of the code:
- The runner shows the flow; the class holds the methods.
- One job = one function / method. Don't split one job across two functions; don't keep a wrapper that only re-routes calls.
- Names say what the thing does, in the user's terms. A method name must not collide with an attribute (`self.zone_centroids` vs `zone_centroids()`).
- New plots: `functions/plot_results.py` (headless `Figure`); GUI canvas: `classes/mpl_canvas.py`.

## What NOT to change

- Behavior, numbers, export filenames / column names, log text used downstream.
- Paper scripts (`docs/paper/**`): leave their imports; list the ones a move breaks.
- No speed-ups or new features; list them as suggestions.

## Verification

- Import-check every importer of a moved name: `.venv\Scripts\python.exe -c "import <module>"` (one command per call).
- Run every pipeline that executes the moved code, test output under `output/test*` (never `results/`), foreground:
  - spontaneous: `.venv\Scripts\python.exe SRE_analysis.py --proc_list <proc list> --results_dir <dir>`
  - permutation: `.venv\Scripts\python.exe permutation_test.py --proc_list <proc list> --stems <one stem> --max_new 5 --out_root <dir>`
  - spike-aligned: `.venv\Scripts\python.exe ach_domain_analysis.py --ana_list <ana list>`
- Compare against the numbers / files of the run before the move (see `neat-refactor` "Verification" for what to compare).

## Report format

- Moves made: `name: old file -> new file`, importers updated
- Paper scripts broken by the moves (not touched)
- `ruff check`: clean / fixed
- Import checks + runs: identical, or the first difference; full output paths
