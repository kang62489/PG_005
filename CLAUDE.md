# PG_005 - Acetylcholine Imaging Analysis
The project is consist of two main parts: image preprocessing (detrend, gaussian filtering) and spike aligned analysis (spike detection, spatial categorization, region analysis). Now I want to create a GUI by python and PySide6 for properly managing the whole analysis pipeline and database created in PG_003 expdata_builder.

## Documentation
1. When create new markdown files (except README.md), always save them to the `docs/` folder. If `docs/` does not exist, create one.
2. Always put the markdown plans created by plan mode in the `.claude/plans/` folder. If `plans/` does not exist, create one.

## Answering Questions
1. Try using step-by-step approach to answer the questions.
2. Try using simple examples with actual data/numbers for explanation.
3. Use "-" or "=" to create separation lines for separating different points/sessions.
4. Use emoji to make the text lively.


### Code Editing
1. Consider impact on both CPU and GPU pipelines if changing preprocessing
2. Check and fix ruff problems of **python files** after editing according to the settings in `pyproject.toml`.

### Adding New Features
- New analysis methods → add to `classes/`
- New processing functions → add to `functions/` with both CPU/GPU if applicable
- New plots → extend `classes/plot_results.py`

## Other advices
*Use emoji to make the text lively. *
*Use obvious characters or symbols or lines (only the separators) to separate sections or paragraphs of your answers. *