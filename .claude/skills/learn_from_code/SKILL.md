---
name: learn_from_code
description: When user ask for saving the conversations about concepts, terms, and principles from our conversations into a knowledge base, usually have keywords like "save this", "remember this", "document this", "knowledge base", "concept", "term", "principle", "summarize", etc.
---

## Example Scenarios
When user asks questions about concepts like:
- How does Gaussian filtering work?
- What's the difference between Otsu and Li thresholding?
- Why use median instead of mean?
- Explain z-score normalization

**And** user wants to save the explanation for later.

## What this skill does

1. Show that this skill is triggered.
2. **Looks for existing notes** in `docs/knowledgebase/` to see if we've talked about this before
3. **Merges with existing notes** if the topic already exists, or creates a new file if it's new
4. **Organizes your questions and my answers** into clean markdown files
5. **Adds keywords** so you can find stuff easily later

## Where files go

**Save to**: `docs/knowledgebase/{topic_name}.md`

**File naming** (use snake_case):
- `gaussian_filtering.md` ✓
- `zscore_normalization.md` ✓
- `image_processing.md` ✗ (too broad, be specific!)

## How to organize content

First, check what's already there:
- Look in `docs/knowledgebase/*.md` for similar topics
- If you find something similar, add to that file instead of making a new one
- If it's totally new, make a new file

## Markdown format to use

```markdown
---
keywords: keyword1, keyword2, keyword3, keyword4
related: other_topic.md (if any)
---

# 2026-01-30 (newest stuff goes here)

## What is Gaussian filtering?

Gaussian filtering is a way to smooth images by...

### Why use it?
- Removes noise
- Preserves important features

### Example
\`\`\`python
# code example if helpful
\`\`\`

---

# 2026-01-15 (older stuff below)

## Previous question about filtering

Previous answer...
```

## Quick checklist

- [ ] Check `docs/knowledgebase/*.md` for similar topics first
- [ ] Use descriptive file names (be specific!)
- [ ] Put newest content at the top
- [ ] Add good keywords for searching later
- [ ] Include examples or formulas if they help
- [ ] Link to related code files if relevant

## Example

Good file names:
- `spike_aligned_median.md` ✓
- `otsu_thresholding_algorithm.md` ✓
- `why_use_sigma_6_gaussian.md` ✓

Too vague:
- `statistics.md` ✗
- `concepts.md` ✗

## Before saving

Show user:
1. The filename you're going to use
2. A quick summary of what's being saved
3. Ask if user wants to merge with an existing file or create new