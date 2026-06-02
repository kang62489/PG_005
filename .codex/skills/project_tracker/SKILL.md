---
name: project_tracker
description: Create a markdown file to log project progress and TODOs when the user is stopping work for a while, or use docs/continue_from_here.md when the user asks where to continue.
---

## Example Scenarios

Use this skill when the user says:

- I am going to take a break now.
- I am going to sleep now.
- I am going to stop working for today.
- I am done for today.
- Can you wrap up what we have done today?

Also use it when the user asks:

- Where were we last time?
- Where should I continue?
- What should I do next?

## What This Skill Does

1. State briefly that the project tracker workflow is being used.
2. Check modified files and summarize the changes.
3. Check current working plans, if any, and summarize progress.
4. List candidate TODOs based on the summaries.
5. Log the last working file name and line number when it is clear.
6. Ask whether extra messages or TODOs should be added only if needed.
7. Summarize the above into a reporting section in the markdown file.
8. Update `docs/continue_from_here.md` accordingly.
9. Show the user the important contents of `docs/continue_from_here.md`.

## Where Files Go

Save to: `docs/continue_from_here.md`

## Markdown Format

```markdown
# Log of the project progress 2026-02-28 Sat 10:00:00

Last working file: <file_name>
Last working line: <line_number>

## List of Modified Files

- classes/helper_combo_editor.py
- classes/model_dynamic_list.py
- controllers/ctrl_rec_import.py (<- Break here, line 123)

## Summary of Current Progress

- We have done 1.
- We have done 2.
- We have done 3.

## Completed TODOs/Tasks

- TODO 1
- TODO 2
- TODO 3

## What Should We Do Next?

- TODO 1
- TODO 2
- TODO 3
```
