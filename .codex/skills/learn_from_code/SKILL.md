---
name: learn_from_code
description: Save conversations about concepts, terms, and principles into a knowledge base when the user says things like "save this", "remember this", "document this", "knowledge base", "concept", "term", "principle", or "summarize".
---

## Example Scenarios

Use this skill when the user asks about concepts such as:

- How does Gaussian filtering work?
- What is the difference between Otsu and Li thresholding?
- Why use median instead of mean?
- Explain z-score normalization.

And the user wants to save the explanation for later.

## What This Skill Does

1. State briefly that the knowledge base workflow is being used.
2. Look for existing notes in `docs/knowledgebase/` to see if the topic already exists.
3. Merge with existing notes if the topic already exists, or create a new file if it is new.
4. Organize the user's questions and the answers into clean markdown.
5. Add keywords so the topic is easy to find later.

## Where Files Go

Save to: `docs/knowledgebase/{topic_name}.md`

File naming uses snake_case:

- `gaussian_filtering.md`
- `zscore_normalization.md`
- Avoid broad names like `image_processing.md`.

## How To Organize Content

First, check what is already there:

- Look in `docs/knowledgebase/*.md` for similar topics.
- If you find something similar, add to that file instead of making a new one.
- If it is totally new, make a new file.

## Markdown Format

```markdown
---
keywords: keyword1, keyword2, keyword3, keyword4
related: other_topic.md
---

# 2026-01-30

## What is Gaussian filtering?

Gaussian filtering is a way to smooth images by...

### Why use it?

- Removes noise
- Preserves important features

### Example

```python
# code example if helpful
```

---

# 2026-01-15

## Previous question about filtering

Previous answer...
```

## Quick Checklist

- Check `docs/knowledgebase/*.md` for similar topics first.
- Use descriptive file names.
- Put newest content at the top.
- Add useful keywords for searching later.
- Include examples or formulas if they help.
- Link to related code files if relevant.

## Before Saving

Show the user:

1. The filename to use.
2. A quick summary of what is being saved.
3. Whether the content should merge with an existing file or create a new one, if that is not clear.
