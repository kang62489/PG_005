---
name: neat-refactor
description: Behavior-identical neatness refactor of a PG_005 Python file (TODO F) -- step-list module docstring, CONFIG / STEP banner blocks, short docstrings, no history commentary -- verified identical on real data. Use when the user says "neaten", "tidy up", "clean up the layout", "neatness refactor", "TODO F", or names a file to make readable without changing behavior.
---

## What this skill does

Rewrites the layout of one file at a time so it reads like `classes/sp_zone_analyzer.py` (the reference style). **Behavior must not change:** same inputs give byte-identical outputs.

1. Show that this skill is triggered.
2. Ask which file(s) to do, or propose an order if the user doesn't name one.
3. For **one** file: read it and list the planned changes (see "What to change" below). **Wait for approval before editing.**
4. Take a reference run **before** editing (see "Verification").
5. Apply the changes. Only layout, docstrings, comments, and ordering within a block.
6. Run `ruff check <file>` (not `uv run ruff`) and fix what it reports.
7. Re-run and compare against the reference. Report "identical" or show the first difference.
8. State the full output paths, then move on to the next file only after the user confirms.

## What to change

| Item | Target |
|---|---|
| Module docstring | A 4-6 line step list: `Step 1. Name : what it does` (see `classes/sp_zone_analyzer.py` lines 1-12). Add an `Example:` block only if the file is a class/function module |
| Imports | `## Modules` header, then `# Standard library imports`, `# Third-party imports`, `# Local imports` groups |
| Constants | One `CONFIG` banner block near the top, grouped by step with `# --- Step N: name ---` sub-headers, each constant with a short trailing comment and unit |
| Banners | `# ====...` blocks with `#   CONFIG` / `#   STEP N -- NAME` titles; sub-blocks `# --- Na. name ---` |
| Functions | Docstrings 1-3 lines; `Args:` / `Returns:` only when a name isn't self-explanatory |
| Comments | Keep *why* comments. Remove history-style ones ("used to...", "moved here because...", "Session 58...") |
| Type hints | Follow `CLAUDE.md`: always annotate return types; annotate arguments only with types already imported for body use; skip Qt override argument annotations |

## What NOT to change

- Logic, numbers, defaults, argument order, public names, return values, file/column names, log text used downstream.
- No renames, new options, new helpers, speed-ups, or "while I'm here" fixes. If one looks worthwhile, **list it as a suggestion and ask**. Don't build it.
- No reordering of statements that could change execution order (e.g. imports with side effects, `check_cuda()` before numba imports).

## Verification (every file)

Pick the run that exercises the file:

| File touched | Run |
|---|---|
| spike-aligned pipeline (`ach_domain_analysis.py`, `classes/region_analyzer.py`, `classes/spike_reliability.py`, `classes/spatial_categorization.py`, `classes/abf_clip.py`, `functions/plot_results.py`, `functions/hotspot_flow.py`, ...) | `.venv/Scripts/python.exe ach_domain_analysis.py --ana_list <ana list>` |
| spontaneous (`spontaneous_analysis.py`, `classes/sp_zone_analyzer.py`, `functions/fit_hist.py`, `functions/zone_kernels.py`) | `.venv/Scripts/python.exe spontaneous_analysis.py --proc_list <proc list> --results_dir <dir>` |

- The reference and after runs go to `output/test*/before` and `output/test*/after` (a new free test number), **never** `results/`. Run in the foreground; the user watches.
- Compare: MED / CAT / ZONE_MASK / ZONE_MAPS TIFFs (`np.array_equal`), `ZONES.npz` arrays, `results.db` rows (skip `timestamp`), and PNGs pixel-wise where titles didn't change. Reuse or extend `output/test5/compare_test5.py`.
- One shell command per tool call, no `;` `&&` `|` chains (see `CLAUDE.md`).

## Report format

For each file:
- Changes made (short list)
- `ruff check`: clean / fixed
- Comparison: identical, or the first mismatch
- Output paths of the before/after runs
