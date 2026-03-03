---
name: quick_code_trace
description: Create markdown files for quickly tracing the code when user asks questions like "Which file is used to create the database?", "Where are the codes related to the saving functions", "Where are the codes related to the plotting of the figure XX?", etc.
---

## Example Scenarios
When user asking questions related to find the code:
- Which file is used to create the database?
- Where are the codes related to the saving functions
- Where are the codes related to the plotting of the figure XX?

And user wants to save a indexing markdown file for quickly tracing the code in the future.

## What this skill does
1. Show that this skill is triggered.
2. Check the code base and find the files related to the question
3. Index the files and the related functions
4. Save the indexing file to `docs/` folder for future reference
5. Answer the question with the indexing file

## Where files go

**Save to**: `docs/quick_code_trace/{question_name}.md`

## Markdown format to use

**Tip**: If possible, use codeblocks to show extracted codes for quick reference.

```markdown
# {Question/Topic Title}

> 📅 Created: {YYYY-MM-DD}
> 🔍 Query: "{Original user question}"

---

## 📁 Files Overview

- `path/to/file1.py` - Brief description
- `path/to/file2.py` - Brief description

---

## 🔗 Code Trace

### 1. `path/to/file1.py`

**Purpose**: What this file does

**Key Functions/Classes**:

- `function_name()` (L42) - What it does
- `ClassName` (L100) - What it represents

**Code Excerpt**:

```python
def function_name(param1, param2):
    """Docstring explaining what this does."""
    result = some_operation()
    return result
```

---

### 2. `path/to/file2.py`

**Purpose**: What this file does

**Key Functions/Classes**:

- `another_function()` (L15) - What it does

**Code Excerpt**:

```python
def another_function():
    """Brief description."""
    ...
```

---

## 📝 Notes

- Additional context or important details
- Related files not directly involved but useful to know
```