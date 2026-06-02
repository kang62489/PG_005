# PG_005 - Acetylcholine Imaging Analysis

This project has two main parts: image preprocessing, including detrending and Gaussian filtering, and spike-aligned analysis, including spike detection, spatial categorization, and region analysis.

The current goal is to build a Python/PySide6 GUI for managing the full analysis pipeline and the database created in PG_003 `expdata_builder`.

## Documentation

1. When creating new markdown files, except `README.md`, save them under `docs/`. Create `docs/` if it does not exist.
2. Save implementation plans under `.codex/plans/`. Create `.codex/plans/` if it does not exist.
3. Keep project notes practical and tied to the actual code paths in this repository.

## Answering Questions

1. Use a step-by-step approach when explaining code, data flow, or debugging.
2. Prefer simple examples with actual data or numbers when they make the explanation clearer.
3. Answer the user's question before proposing code changes.
4. Keep responses readable with short sections or blocks when the answer is more than a few lines.

## Code Editing

1. Do confirm with me before any modification
2. After editing Python files, check and fix Ruff problems according to `pyproject.toml`.

## Type Annotation Rules

- Always annotate return types on every function/method.
- Argument annotations: only annotate if the type is already imported for use in the function body. Never add an extra import solely for annotating an argument.
- Qt override methods such as `data`, `flags`, `setData`, `createEditor`, and `eventFilter`: skip argument annotations unless the type is already imported for other use in the file.
- Good example: `def load(self, path: Path) -> pl.DataFrame` because `Path` and `pl.DataFrame` are already imported for body use.
- Bad example: `def flags(self, index: QModelIndex) -> Qt.ItemFlag` if `QModelIndex` is imported only for this annotation.

## Adding New Features

- New analysis methods: add to `classes/`.
- New processing functions: add to `functions/` with both CPU/GPU if applicable.
- New plots: extend `classes/plot_results.py`.

## Local Codex Notes

- The original Claude configuration is preserved in `.claude/` and `CLAUDE.md`.
- Project-local Codex notes are stored in `.codex/`.
- The main repo instruction file for Codex is this `AGENTS.md`.
- Reusable Codex skills may also be installed globally under `C:\Users\KANG\.codex\skills\` if you want them available across projects.

## Important
- Answering user's questions before proposing any code modification
