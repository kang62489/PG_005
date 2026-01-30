# PG_005 - Acetylcholine Imaging Analysis
1. Check `doc/PROJECT_SUMMARY.md` and `doc/DEPENDENCY_DIAGRAM.md` for codebase review

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