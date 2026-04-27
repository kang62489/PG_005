---
keywords: polars, dataframe, expressions, pl.lit, pl.col, pl.all, pl.struct, select, with_columns, cast, alias, map_elements, iter_rows, str.split, list.first, list.last, list.get, clone, immutable
related: pick_list_dialog_mechanism.md
---

# 2026-04-26

## Polars Basics — DataFrame Structure

Polars DataFrames have **column headers (x-axis) but no row index (y-axis)**. Unlike pandas, rows have no labels — you refer to them by condition or position only.

```
Filename             | Count
---------------------|------
20230101-rec001.tif  | 5
20230102-rec002.tif  | 3
```

---

## `pl.lit()` — Fixed value for every row

Wraps a plain Python value so Polars can use it as a column expression. Every row gets the same value.

```python
df.with_columns(pl.lit("OIST").alias("lab"))
```
```
name  | lab
------|-----
Alice | OIST
Bob   | OIST
```

> Use `pl.col("name")` for existing columns, `pl.lit(value)` for fixed constants.

---

## `with_columns()` — Add or overwrite columns, keep originals

Returns the same table with new/replaced columns added. Original columns are preserved.

Start with:
```
Filename             | Size
---------------------|-----
20230101-rec001.tif  | 100
20230102-rec002.tif  | 200
```

```python
df.with_columns(pl.lit("OIST").alias("lab"))
```

Result:
```
Filename             | Size | lab
---------------------|------|-----
20230101-rec001.tif  | 100  | OIST
20230102-rec002.tif  | 200  | OIST
```

`Filename` and `Size` are still there. `lab` was added on top.

You can also **overwrite an existing column** by using the same name in `.alias()`:

```python
df.with_columns(pl.col("Size").cast(pl.Utf8).alias("Size"))
```

Result:
```
Filename             | Size   ← now a string, not int
---------------------|------
20230101-rec001.tif  | "100"
20230102-rec002.tif  | "200"
```

---

## `select()` — Build a new table, originals gone

Only the columns you specify survive. Original columns are discarded.

Start with the same DataFrame:
```
Filename             | Size
---------------------|-----
20230101-rec001.tif  | 100
20230102-rec002.tif  | 200
```

```python
df.select(
    pl.col("Filename").str.split("-").list.first().alias("DOR"),
    pl.lit("").alias("IMG_READY"),
)
```

Result:
```
DOR        | IMG_READY
-----------|----------
20230101   |
20230102   |
```

`Filename` and `Size` are gone — only `DOR` and `IMG_READY` remain.

| | `with_columns()` | `select()` |
|---|---|---|
| Keeps originals? | ✅ Yes | ❌ No |
| Use when | adding/updating columns | building a completely new table |

---

## `pl.all()` — Select every column

Shorthand for selecting all columns at once.

```python
df.with_columns(pl.all().cast(pl.Utf8))
# Converts every column to string type
```

---

## `.cast()` — Convert data type

Converts a column to a different type. `pl.Utf8` is Polars' name for string/text.

```python
pl.col("Count").cast(pl.Utf8)   # int → string
pl.all().cast(pl.Utf8)          # all columns → string
```

> Not related to file encoding like `encoding="UTF-8"`. This is purely about the DataFrame column type.

---

## `.alias()` — Rename the result column

Gives the output of an expression a new column name. Without it, Polars reuses the original column name (confusing).

```python
pl.col("Filename").str.split("-").list.first().alias("DOR")
# Result column is named "DOR", not "Filename"
```

---

## `is_empty()` — Check if DataFrame has zero rows

```python
if df.is_empty():
    # no rows
```

---

## String splitting — `str.split()` + `list` accessors

Split a string column and extract parts by position.

```python
# "20230101-rec001.tif" → ["20230101", "rec001.tif"]
pl.col("Filename").str.split("-")
```

| What you want | Code | Result |
|---|---|---|
| First part | `.list.first()` | `"20230101"` |
| Last part | `.list.last()` | `"rec001.tif"` |
| Specific index | `.list.get(1)` | `"rec001.tif"` (index 0-based) |

```python
# Full example: extract DOR and TIFF_SERIAL from "20230101-rec001.tif"
pl.col("Filename").str.split("-").list.first().alias("DOR")           # "20230101"
pl.col("Filename").str.split("-").list.last()
    .str.replace(r"\.tif$", "").alias("TIFF_SERIAL")                  # "rec001"
```

---

## `pl.struct()` + `map_elements()` — Row-wise function on multiple columns

When you need to apply a Python function that uses **more than one column**, bundle them with `pl.struct()` first. Each row is passed as a dict.

```python
# Single column
pl.col("name").map_elements(lambda x: x.upper(), return_dtype=pl.Utf8)

# Multiple columns
pl.struct(["first", "last"]).map_elements(
    lambda r: f"{r['first']} {r['last']}",
    return_dtype=pl.Utf8,
).alias("full_name")
```

**Real usage — check file existence per row:**
```python
df.with_columns(
    pl.struct(["DOR", "TIFF_SERIAL"]).map_elements(
        lambda r: "Yes" if (raw_dir / f"{r['DOR']}-{r['TIFF_SERIAL']}.tif").exists() else "No",
        return_dtype=pl.Utf8,
    ).alias("IMG_READY"),
)
```

> 🧠 Rule: one column → `pl.col().map_elements()`, multiple columns → `pl.struct([...]).map_elements()`

---

## `iter_rows()` — Iterate row by row (use sparingly)

Polars is designed for vectorized column operations. Row-by-row iteration is slow and discouraged. Use `map_elements()` instead when possible.

If you must iterate:
```python
for row in df.iter_rows(named=True):  # named=True → row is a dict
    print(row["DOR"])
```

> `iter_rows()` without `named=True` returns tuples — can't access by column name.

---

## `.clone()` — Usually unnecessary in Polars

Polars DataFrames are **immutable** — operations like `with_columns()` and `select()` always return a **new DataFrame** without modifying the original.

```python
# ❌ Unnecessary — original is never modified anyway
file_check_list = self.model_pick_list._data.clone()

# ✅ Safe to do directly
file_check_list = self.model_pick_list._data
```

> Only use `.clone()` if you explicitly need a fully independent copy for some external reason. In normal Polars workflows it is never needed.
