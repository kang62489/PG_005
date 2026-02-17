# PG_005 - Acetylcholine Imaging Analysis
1. Check `doc/PROJECT_SUMMARY.md` and `doc/DEPENDENCY_DIAGRAM.md` for understanding the project and getting into situation.
2. Check `.cloud/plans` for understanding the current progress. Update the plan if necessary.
3. Check and fix ruff problems in **python files** if pyproject.toml has relative settings.
4. Check what skills you have.
5. Answer the questions on the basis of your understanding of this project.
6. Use code block if necessary, especially for visualization.

### Before refactoring
1. Always make a todo.md at the root of the project for us to track the progress.
2. Update `doc/PROJECT_SUMMARY.md` and `doc/DEPENDENCY_DIAGRAM.md` accordingly.

### Before Modifying Analysis Code
1. Consider impact on both CPU and GPU pipelines if changing preprocessing

### Adding New Features
- New analysis methods → add to `classes/`
- New processing functions → add to `functions/` with both CPU/GPU if applicable
- New plots → extend `classes/plot_results.py`

*When in doubt, check the docs. They explain the "why" behind implementation choices.*
*Use emoji to make the text lively. *
*Use obvious characters or symbols or lines (only the separators) to separate sections or paragraphs of your answers. *