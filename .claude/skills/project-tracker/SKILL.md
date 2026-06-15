---
name: project-tracker
description: Create a markdown file to log and track the project's current progress and TODOs when user expresses that he/she is going to interrupt current work for a while (keywords such as "break", "sleep", "stop", "done", "wrap up", "food", "lunch", "dinner", "meal" may appear in the conversation); remind user where to continue based on docs/continue_from_here.md when user ask "where were we left last time?", "where should I continue?", "what should I do next?", etc.
---

## Example Scenarios

### Wrap-up (full session log)
User says:
- I'm going to take a break now
- I'm going to sleep now
- I'm going to stop working for today
- I'm done for today
- can you wrap up what we have done today

### Recap (concise summary only)
User says:
- recap
- give me a recap
- what did we do today
- summarize what we did


## What this skill does

### For wrap-up triggers
1. Show that this skill is triggered.
2. Check modified files and then summarize the changes
3. Check current working plans (if any) and then summarize the progress
4. Ask user to confirm TODOs via **interactive multi-select (AskUserQuestion)**
   - Derive TODOs from TODAY'S actual changes only — never copy carry-overs from old session logs blindly
   - Think: "what is unfinished, what has a side effect elsewhere, what might break downstream?"
5. Log the last working file name and line number
6. Ask user if any extra notes or TODOs need to be added
7. Write everything to `docs/continue_from_here.md`, including a `## Last Session Recap` line:
   ```
   ※ recap: <1-2 sentences: what was worked on, what was fixed/added, and whether anything is pending>
   ```
8. Print the contents of `docs/continue_from_here.md` to the terminal

### For recap triggers
1. First check if conversation context is available (i.e. this is not a fresh/cleared session)
   - **If context exists**: generate the recap line directly from the conversation
   - **If no context** (fresh session): read `docs/continue_from_here.md` and use the `## Last Session Recap` line saved there
2. Output the recap line in this exact format:
   ```
   ※ recap: <1-2 sentences: what was worked on, what was fixed/added, and whether anything is pending>
   ```
   - Be specific — name the feature/bug/file, not generic descriptions
   - Keep it under 40 words
   - No file writing, no TODO prompts — just the recap line

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

## Last Session Recap
※ recap: <concise 1-2 sentence summary of what was done and what is pending>

```