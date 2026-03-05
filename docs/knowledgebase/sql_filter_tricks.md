---
keywords: SQL, WHERE clause, filter, 1=0, boolean, QSqlTableModel, setFilter, show nothing
related: spatial_frequency_filtering.md
---

# 2026-03-05

## The `1=0` Trick — Force SQL to Return No Rows

### Context
In `controllers/ctrl_dor_query.py`, `apply_filters()` uses `model.setFilter()` to filter
`tv_rec_summary` based on which items are checked in the `CheckableDropdown` widgets.
When the user unchecks **all items** in a dropdown, we want to show **zero rows**.

### The Problem
`QSqlTableModel.setFilter()` expects a SQL string (a WHERE clause).
- `setFilter("")` → removes filter → **shows all rows** ❌
- `setFilter(False)` → type error, expects string ❌
- We need something that means "match nothing" ✅

### The Trick: `1=0`

```python
model.setFilter("1=0")
```

This produces:
```sql
SELECT * FROM REC_dor WHERE 1=0
```

SQL evaluates `1=0` as a boolean for **every row** — and since `1` never equals `0`,
every row fails the check → **zero rows returned**.

### Why not `WHERE FALSE`?
SQLite does not support `WHERE FALSE` directly. `1=0` is the portable workaround
that works across all SQL databases.

### Python analogy
```python
if 1 == 0:   # never runs
    do_something()
```
Same idea — a condition that is always false.

### How it fits into `apply_filters()`

```python
if len(checked) == 0:
    conditions.append("1=0")  # append the trick directly
    break                      # skip remaining columns — pointless to check further
```

The `break` ensures we exit the loop immediately. Then:
```python
model.setFilter(" AND ".join(conditions))
# → model.setFilter("1=0")
```

`join` on a single-element list returns the element as-is (no separator added).

### Filter Logic Summary

| Scenario | SQL condition |
|---|---|
| All items checked | *(column skipped, no condition)* |
| Some items checked | `col IN ('val1', 'val2', ...)` |
| No items checked | `1=0` |
| Multiple columns | combined with `AND` |
