---
name: project_tracker
description: Create a markdown file to log and track the project's current progress and TODOs when user expresses that he/she is going to interrupt current work for a while (keywords such as "break", "sleep", "stop", "done", "wrap up", "food", "lunch", "dinner", "meal" may appear in the conversation); remind user where to continue based on docs/continue_from_here.md when user ask "where were we left last time?", "where should I continue?", "what should I do next?", etc.
---

## Example Scenarios
User says:
- I'm going to take a break now
- I'm going to sleep now
- I'm going to stop working for today
- I'm done for today
- can you wrap up what we have done today


## What this skill does
1. Show that this skill is triggered.
2. Check modified files and then summarize the changes
3. Check current working plans (if any) and then summarize the progress
4. List candidates of TODOs based on the summaries, ask user to choose
5. Log the last working file name and line number
6. Ask user to see if any extra messages or todos need to be added
7. Summarize above into a reporting sections in the markdown file
8. Update the ``docs/continue_from_here.md`` file accordingly
9. Print the contents of `doc/continue_from_here.md` file to the terminal

## Where files go

**Save to**: `docs/continue_from_here.md`

## Markdown format to use

```markdown
# Log of the project progress 2026-02-28 Sat 10:00:00
Last working file: <file_name>
Last working line: <line_number>

# List of modified files:
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

## Summary of current progress (based on modified files, existing plans)
- We have done 1
- We have done 2
- We have done 3

## Completed TODOs/Tasks (before new wrap-up)
- TODO 1
- TODO 2
- TODO 3

## What should we do next? (TODOs)
- TODO 1
- TODO 2
- TODO 3

```