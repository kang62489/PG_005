---
name: project_tracker
description: Create a markdown file for user to log and track the project's current progress and TODOs
---

## When to use this skill

When user expresses that he/she is going to interrupt current work for a while (keywords such as "break", "sleep", "stop", "done" may appear in the conversation):
- I'm going to take a break now
- I'm going to sleep now
- I'm going to stop working for today
- I'm done for today
- can you wrap up what we have done today


## What this skill does

1. Check current working plan (if any)
2. Check current TODOs (if any)
3. Check current working files
4. Ask user for leaving messages for what should continue next time
5. Summarize above into a reporting sections in the log file
6. Sumarize 1 - 4 into a TODO list in the log file

## Where files go

**Save to**: `continue_from_here.md`

## Markdown format to use

```markdown
# Log of the project progress 2026-02-28 Sat 10:00:00
List of modified files:
- classes/helper_combo_editor.py
- classes/model_dynamic_list.py
- classes/model_metadata_form.py
- classes/thread_tiff_stacker.py
- controllers/ctrl_abf_note.py
- controllers/ctrl_exp_info.py
- controllers/ctrl_rec_import.py (<- Break here, line 123)
- controllers/ctrl_rec_writer.py
- controllers/ctrl_tiff_stacker.py
- main.py
- styles/styles.qss
- ui/ui_mainwindow.ui

## What have we done? (Summary of current progress)
- We have done 1
- We have done 2
- We have done 3

## What should we do next? (TODOs)
- TODO 1
- TODO 2
- TODO 3

```