# Python `re` Module — Regex Symbol Cheatsheet

## Basic Symbols

| Symbol | Meaning | Example pattern | Matches | Does NOT match |
|--------|---------|-----------------|---------|----------------|
| `.`    | Any single character (except newline) | `a.c` | `abc`, `a1c`, `a-c` | `ac`, `abbc` |
| `*`    | 0 or more of the preceding | `ab*c` | `ac`, `abc`, `abbc` | `adc` |
| `+`    | 1 or more of the preceding | `ab+c` | `abc`, `abbc` | `ac` |
| `?`    | 0 or 1 of the preceding (also: non-greedy modifier) | `ab?c` | `ac`, `abc` | `abbc` |
| `\`    | Escape — treat next char as literal | `\[` | `[` | anything else |

---

## Greedy vs Non-greedy

By default `+` and `*` are **greedy** — they grab as much as possible.
Adding `?` after them makes them **non-greedy** — stop at the first match.

```
Input:  "[hello, world]"
(.+)    →  matches  "hello, world]"   (greedy: goes to the end)
(.+?)   →  matches  "hello"           (non-greedy: stops at first comma)
```

---

## Character Classes `[...]`

| Symbol | Meaning | Example | Matches |
|--------|---------|---------|---------|
| `[abc]` | any one of a, b, c | `[abc]+` | `cab`, `ba` |
| `[^abc]` | any char EXCEPT a, b, c | `[^,]+` | anything without a comma |
| `[a-z]` | any lowercase letter | `[a-z]+` | `hello` |
| `\w` | word char: `[a-zA-Z0-9_]` | `\w+` | `BIEXP`, `hello_2` |
| `\s` | whitespace (space, tab, etc.) | `\s*` | ` `, `   `, `` (empty) |
| `\d` | digit: `[0-9]` | `\d+` | `123`, `42` |

---

## Groups `(...)`

| Syntax | Meaning | Example |
|--------|---------|---------|
| `(abc)` | capturing group — result accessible via `.group(n)` | `(hello)` |
| `(?:abc)` | non-capturing group — groups for structure but not stored | `(?:hello)?` |

```python
m = re.match(r"(\w+),\s*(\w+)", "BIEXP, SKIP")
m.group(1)  # "BIEXP"
m.group(2)  # "SKIP"
```

---

## Real Example from `img_proc.py`

Pattern: `r"\[(.+?),\s*([^,]+?),\s*(\w+),\s*(\w+)(?:,\s*(.+?))?\]"`

Input: `[2026_03_20-0028.tif, BIEXP, SKIP, BIEXP, 2026_03_20_0015.abf]`

| Part | Pattern | Captures |
|------|---------|----------|
| `\[` | literal `[` | — |
| `(.+?)` | filename (non-greedy) | `2026_03_20-0028.tif` |
| `,\s*` | comma + optional spaces | — |
| `([^,]+?)` | gauss_exists (no commas allowed, handles spaces like "BIEXP & MOV") | `BIEXP` |
| `,\s*` | comma + optional spaces | — |
| `(\w+)` | do_processing (word chars only) | `SKIP` |
| `,\s*` | comma + optional spaces | — |
| `(\w+)` | detrend_mode (word chars only) | `BIEXP` |
| `(?:,\s*(.+?))?` | optional paired_abf (whole group optional) | `2026_03_20_0015.abf` |
| `\]` | literal `]` | — |
