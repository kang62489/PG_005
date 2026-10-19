---
name: neat-refactor
description: Behavior-identical neatness refactor of a PG_005 Python file -- numbered flow in the module docstring, CONFIG + named banner blocks (no step numbers), commented runner flow, short docstrings, no history commentary -- verified identical on real data (moving code between files is the file-structure skill). Use when the user says "neaten", "tidy up", "clean up the layout", "neatness refactor", "TODO F", or names a file to make readable without changing behavior.
---

## What this skill does

Rewrites the layout of one file at a time so it reads like the templates below. **Behavior must not change:** same inputs give byte-identical outputs.

1. Show that this skill is triggered.
2. Ask which file(s) to do, or propose an order if the user doesn't name one.
3. For **one** file: read it and list the planned changes (see "What to change"), plus any structure issues as a separate list (see "Structure"). **Wait for approval before editing.**
4. Take a reference run **before** editing (see "Verification").
5. Apply the approved changes.
6. Run `ruff check <file>` (not `uv run ruff`) and fix what it reports.
7. Re-run and compare against the reference. Report "identical" or show the first difference.
8. State the full output paths, then move on to the next file only after the user confirms.

## Templates

| File kind | Template |
|---|---|
| Runner script (flow of a pipeline) | `SRE_analysis.py`, `permutation_test.py` |
| Analysis class | `classes/sre_analyzer.py` (pipeline class), `classes/hierarchical_clustering.py` (one self-contained job) |
| Processing functions (CPU / GPU) | `functions/flash_mask.py`, `functions/footprint_traces.py`, `functions/permutation.py` |

## What to change

### All files

| Item | Target |
|---|---|
| Module docstring | Title line, then a **numbered** flow list `Step 1. Name : what it does` (numbers are fine here: it is the flow). Then short notes, then `Usage:` (runner) or `Example:` (class / function module) |
| Imports | `## Modules` header, then `# Standard library imports`, `# Third-party imports`, `# Local imports` groups. Keep side-effect order (e.g. `check_cuda()` before anything imports numba) |
| CONFIG | One `CONFIG` banner block near the top; constants grouped with `# --- name ---` sub-headers (name of the block / method that uses them), each constant with a short trailing comment and unit |
| Banners | `# ====...` blocks titled by **what the block does** (`#   FILL + FILTER`, `#   EXPORT related local functions`). **No step numbers** in banners or sub-blocks (`# --- CPU (numba, one frame per thread) ---`, not `# --- 1a. ... ---`): steps jump and repeat in real code |
| Step references | Outside the module docstring, never `Step N` / `Steps 1-5`: use the stage name (`# Detection`, `"""Cutoff: ..."""`, import tag `# export`) |
| Index lines | Under each banner, one `# fn: name, what it does` line per function of the block |
| Functions | Docstrings 1-3 lines; `Args:` / `Returns:` only when a name isn't self-explanatory |
| Comments | Keep *why* comments. Remove history-style ones ("used to...", "moved here because...", "Session 58...") |
| Type hints | Follow `CLAUDE.md`: always annotate return types; annotate arguments only with types already imported for body use; skip Qt override argument annotations |

### Runner script (`SRE_analysis.py`)

| Item | Target |
|---|---|
| Flow | A `FLOW` banner with `run()`. Inside, `# ----` sub-banners named by stage (`# Input`, `# Detection`, ...); setup before the loop marked `(set once, before the loop)` |
| Call comments | Every call in `run()` gets a comment saying what it is and does: `# class method: detect() -- ...`, `# local fn: export_dendrogram() -- ...`, `# imported fn: filelist_parser()`, `# imported class: SREAnalyzer -- ...`. Inside a loop where stages mix, prefix the stage: `# Export. local fn: ...` |
| Local functions | After `FLOW`, one banner per stage in flow order: `#   <STAGE> related local functions`; empty stage -> `# (none yet)`. Unplaced functions -> `#   NOT SORTED YET (-> <STAGE>)` |
| Import tags | Local imports tagged with the stage(s) that use them: `timed,  # detection, export` |

### Analysis class (`classes/sre_analyzer.py`)

| Item | Target |
|---|---|
| Module docstring | `CLASS  Name : method flow` line, `BLOCK  NAME : ...` lines for module-level blocks, where moved-out parts live, `Example:` |
| Class docstring | `Results:` = the attributes each public method sets |
| Sections | In-class `# ----` sub-banners per public method group (`# detect: ...`, `# map: ...`); helpers sit right after the method that calls them |

### Processing functions (`functions/flash_mask.py`)

| Item | Target |
|---|---|
| Blocks | One banner per job; CPU / GPU / Dispatch sub-blocks for kernels |

## Structure

Where code lives (shared code, methods outside their class, cross-file imports) is the `file-structure` skill. If you spot such an issue, list it under "Structure suggestions" and propose running `file-structure`; don't move code across files here.

## What NOT to change

- Logic, numbers, defaults, argument order, public names, return values, file/column names, log text used downstream.
- No renames, new options, new helpers, speed-ups, or "while I'm here" fixes. If one looks worthwhile, **list it as a suggestion and ask**. Don't build it.
- No reordering of statements that could change execution order.

## Verification (every file)

Pick the run that exercises the file:

| File touched | Run |
|---|---|
| spike-aligned pipeline (`ach_domain_analysis.py`, `classes/region_analyzer.py`, `classes/spike_reliability.py`, `classes/spatial_categorization.py`, `classes/abf_clip.py`, `functions/plot_results.py`, `functions/flash_flow.py`, ...) | `.venv/Scripts/python.exe ach_domain_analysis.py --ana_list <ana list>` |
| spontaneous (`SRE_analysis.py`, `classes/sre_analyzer.py`, `classes/hierarchical_clustering.py`, `functions/fit_hist.py`, `functions/flash_mask.py`, `functions/footprint_traces.py`) | `.venv/Scripts/python.exe SRE_analysis.py --proc_list <proc list> --results_dir <dir>` |
| permutation (`permutation_test.py`, `functions/permutation.py`, `functions/flash_mask.py`) | `.venv/Scripts/python.exe permutation_test.py --proc_list <proc list> --stems <one stem> --max_new 5 --out_root <dir>` (seeded lags -> repeatable) |

- The reference and after runs go to `output/test*/before` and `output/test*/after` (a new free test number), **never** `results/`. Run in the foreground; the user watches.
- Compare with a small script in the same test folder:
  - spontaneous: `*_ZONE_MAPS.tif`, `mask/*_FLASH_MASK.tif` (`np.array_equal`), `footprints/*.npz` arrays, `*_ZONES.xlsx` sheets, `clustering/*.png` pixel-wise where titles didn't change
  - spike-aligned: MED / CAT / ZONE_MASK TIFFs, `results.db` rows (skip `timestamp`)
  - permutation: `{stem}_th_flash_sizes.npz`, `{stem}_surrogates.csv` (sur_idx, largest_sur_flash_px)
- One shell command per tool call, no `;` `&&` `|` chains (see `CLAUDE.md`).

## Report format

For each file:
- Changes made (short list)
- Structure suggestions (if any), not applied
- `ruff check`: clean / fixed
- Comparison: identical, or the first mismatch
- Output paths of the before/after runs
