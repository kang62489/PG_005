---
name: trouble_shooter
description: Document bugs and problems that took effort to solve, with their solutions
---

## When to use this skill

When user has just solved a tricky bug or problem

**And** user wants to save what went wrong and how it was fixed.

## What this skill does

1. **Suggests a clear file title** based on the problem for you to approve
2. **Checks if similar issues exist** in `doc/resolved_problems/`
3. **Organizes the debugging history** from our conversation
4. **Creates a clean summary** with problem + solution in markdown

## Where files go

**Save to**: `doc/resolved_problems/{problem_description}.md`

**File naming** (use snake_case, be specific):
- `cuda_out_of_memory_during_gaussian_blur.md` ✓
- `spatial_categorizer_wrong_threshold_values.md` ✓
- `plot_not_showing.md` ✗ (too vague - which plot? why?)
- `buttons_not_working.md` ✗ (too broad, be specific!)

## How to organize content

First, check what's already there:
- Look in `doc/resolved_problems/*.md` for similar issues
- If you find the same bug, add to that file (maybe it came back!)
- If it's a new problem, create a new file

## Markdown format to use

```markdown
---
keywords: keyword1, keyword2, error_message_snippet
files_changed: path/to/file1.py, path/to/file2.py
severity: critical / major / minor
---

# 2026-01-30 (latest fix)

## Problem Description

Brief summary of what went wrong and where it showed up.

### Symptoms
- What the user saw (error messages, wrong behavior, etc.)
- When it happened (specific conditions)

### Example Error (if applicable)
\`\`\`
Paste actual error message here
\`\`\`

## Root Cause

Explain what was actually causing the problem.

## Solution

### Files Changed
- `path/to/file.py:123` - What you changed
- `path/to/another.py:45` - Another change

### Code Changes
\`\`\`python
# Before (broken)
old_code_snippet

# After (fixed)
new_code_snippet
\`\`\`

### Why This Fixes It
Explain why the original code didn't work and why the new code does.

---

# 2026-01-15 (if problem happened before)

## Problem Description

Previous occurrence...
```

## Quick checklist

- [ ] Check `doc/resolved_problems/*.md` first for similar issues
- [ ] Use descriptive file names that explain the actual problem
- [ ] Include the error message or symptoms
- [ ] Explain root cause, not just the fix
- [ ] List which files were changed
- [ ] Add keywords for easy searching later
- [ ] Note severity (helps prioritize if it comes back)

## Before saving

Show user:
1. The filename you're going to use
2. Brief summary of problem + solution
3. Which files were changed
4. Ask if user wants to merge with existing issue or create new

## Example

Good file names:
- `abfclip_spike_detection_off_by_one_error.md` ✓
- `resultsexporter_sqlite_foreign_key_constraint.md` ✓
- `qt_window_freezes_on_large_image_segments.md` ✓

Too vague:
- `bug_fix.md` ✗
- `error_in_processing.md` ✗
- `something_broken.md` ✗