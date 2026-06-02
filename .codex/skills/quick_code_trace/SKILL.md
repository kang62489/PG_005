---
name: quick_code_trace
description: Create markdown files for quickly tracing code when the user asks questions like "Which file creates the database?", "Where is saving implemented?", or "Where is figure plotting implemented?"
---

## Example Scenarios

Use this skill when the user asks questions related to finding code, such as:

- Which file is used to create the database?
- Where are the codes related to the saving functions?
- Where are the codes related to plotting figure XX?

And the user wants to save an indexing markdown file for tracing the code in the future.

## What This Skill Does

1. State briefly that the quick code trace workflow is being used.
2. Check the codebase and find files related to the question.
3. Index the files and related functions/classes.
4. Save the indexing file for future reference.
5. Answer the question and point to the indexing file.

## Where Files Go

Save to: `docs/quick_code_trace/{question_name}.md`

## Markdown Format

If useful, include short code excerpts for quick reference.

```markdown
# {Question/Topic Title}

> Created: {YYYY-MM-DD}
> Query: "{Original user question}"

---

## Files Overview

- `path/to/file1.py` - Brief description
- `path/to/file2.py` - Brief description

---

## Code Trace

### 1. `path/to/file1.py`

Purpose: What this file does

Key Functions/Classes:

- `function_name()` (L42) - What it does
- `ClassName` (L100) - What it represents

Code Excerpt:

```python
def function_name(param1, param2):
    """Docstring explaining what this does."""
    result = some_operation()
    return result
```

---

## Notes

- Additional context or important details
- Related files not directly involved but useful to know
```
