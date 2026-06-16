# PG_005 - Acetylcholine Imaging Analysis
The project is consist of two main parts: image preprocessing (detrend, gaussian filtering) and spike aligned analysis (spike detection, spatial categorization, region analysis). Now I want to create a GUI by python and PySide6 for properly managing the whole analysis pipeline and database created in PG_003 expdata_builder.

## Documentation
1. When create new markdown files (except README.md), always save them to the `docs/` folder. If `docs/` does not exist, create one.
2. Always put the markdown plans created by plan mode in the `.claude/plans/` folder. If `plans/` does not exist, create one.

## Answering Questions
1. Try using step-by-step approach to answer the questions.
2. Try using simple examples with actual data/numbers for explanation.
3. Use `-` or `=` to create separation lines for separating different points/sessions.
4. Use emoji to make the text lively.


### Environment Setup
- When running Python-related shell commands, check if `.venv/Scripts/activate` exists. If it does, prepend activation to the same command so the venv is active for that call:
  - **PowerShell**: `& .venv\Scripts\Activate.ps1; <cmd>`
  - **Bash**: `source .venv/Scripts/activate && <cmd>`

### Code Editing
1. Consider impact on both CPU and GPU pipelines if changing preprocessing
2. Check and fix ruff problems of **python files** after editing according to the settings in `pyproject.toml`.
   - Run ruff directly: `ruff check <files>` (installed via `uv tool install ruff`, shim at `C:\Users\Kang\scoop\persist\uv\tools\shims\ruff.exe`)
   - Do **not** use `uv run ruff` — it triggers package reinstalls even with the venv active, causing file-lock errors when the GUI is running.

### Type Annotation Rules
- **Always** annotate return types on every function/method.
- **Argument annotations** — only annotate if the type is already imported for use in the function body. Never add an extra import solely for annotating an argument.
- **Qt override methods** (e.g. `data`, `flags`, `setData`, `createEditor`, `eventFilter`) — skip argument annotations unless the type is already imported for other use in the file.
- Example ✅: `def load(self, path: Path) -> pl.DataFrame` — `Path` and `pl.DataFrame` already imported for body use.
- Example ❌: `def flags(self, index: QModelIndex) -> Qt.ItemFlag` — if `QModelIndex` is imported only for this annotation, remove it.

### Adding New Features
- New analysis methods → add to `classes/`
- New processing functions → add to `functions/` with both CPU/GPU if applicable
- New plots → extend `classes/plot_results.py`

## Other advices
**Try answering user's questions before proposing any code modification**
**Properly spacing or separating responses/replys into more reader-friendly format, such as sections or blocks**
*Use emoji to make the text lively.*

### Wrap-up / Project Tracker Rules
- **NEVER blindly copy carry-over TODOs from old session logs.** Only include TODOs that are genuinely unfinished based on today's actual changes.
- If a TODO appears in a previous session log, verify it is still relevant and unfinished before including it — do NOT assume it carries over automatically.
- **Always use interactive multi-select (AskUserQuestion) for TODO confirmation** — never just print a list and ask for text confirmation.