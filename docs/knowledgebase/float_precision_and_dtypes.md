---
keywords: float16, float32, float64, floating point, precision, machine epsilon, dtype, significant digits, bit layout, exponent, fraction, mantissa, step size, tiff, image saving
related: spatial_frequency_filtering.md
---

# 2026-05-07

## What is float16/32/64?

The number after "float" is the **total number of bits** used to store the number.

```
float16:  16 bits total
float32:  32 bits total  ("single precision")
float64:  64 bits total  ("double precision")
```

---

## Bit layout (IEEE 754 standard — fixed for all computers)

```
float16:   1 sign  +  5 exponent + 10 fraction  = 16 bits
float32:   1 sign  +  8 exponent + 23 fraction  = 32 bits
float64:   1 sign  + 11 exponent + 52 fraction  = 64 bits
```

- **Sign bit**: + or -
- **Exponent bits**: determines scale (which power-of-2 window)
- **Fraction bits**: determines precision (detail within the window)

---

## How a number is stored

Any number is represented as:

```
value  =  1.fraction  x  2^exponent
```

Like scientific notation but in base 2 instead of base 10.

Example: `123.45`
```
123.45  =  1.9289...  x  2^6     (because 2^6 = 64, 2^7 = 128)

exponent = 6  -> stored in 5 bits as 00110
fraction = 0.9289... -> stored as 10-bit binary series
```

---

## Fraction bits as a binary series

The 10 fraction bits store a number between 1.0 and 2.0 as a series:

```
fraction = 1.  b1     b2     b3     b4   ...   b10
              1/2    1/4    1/8    1/16        1/1024
```

Each bit b is 0 or 1. This is a finite binary series — same concept as decimal expansion but in base 2.

Example: storing 0.9289...
```
bit 1  (1/2)    = 1  ->  0.5000
bit 2  (1/4)    = 1  ->  0.2500
bit 3  (1/8)    = 1  ->  0.1250
bit 4  (1/16)   = 0  ->  0.0000
bit 5  (1/32)   = 1  ->  0.0313
bit 6  (1/64)   = 1  ->  0.0156
bit 7  (1/128)  = 0  ->  0.0000
bit 8  (1/256)  = 1  ->  0.0039
bit 9  (1/512)  = 1  ->  0.0020
bit 10 (1/1024) = 1  ->  0.0010
                  ─────────────
                       0.9287  (close enough to 0.9289)
```

---

## Minimum step size (machine epsilon)

The last fraction bit (1/1024) is the finest detail in the fraction.
When multiplied back by 2^exponent to get the final number, it becomes:

```
minimum step size  =  (1/1024)  x  2^exponent
```

This is what determines how fine the stored details are near any given value.

| Value | Exponent | Step size | Usable decimal places |
|---|---|---|---|
| ~0.5% (dF/F0) | -1 | 1/1024 x 0.5 = 0.0005 | 3-4 places |
| ~5% (dF/F0) | 2 | 1/1024 x 4 = 0.004 | 2-3 places |
| ~123 | 6 | 1/1024 x 64 = 0.0625 | 1-2 places |
| ~1213 (baseline) | 10 | 1/1024 x 1024 = 1.0 | no decimals |
| ~20000 (baseline) | 14 | 1/1024 x 16384 = 16 | jumps of 16! |

---

## Significant digits

The number of significant decimal digits comes from how many distinct values the fraction bits can represent:

```
significant digits  =  fraction_bits  x  log10(2)
                     =  fraction_bits  x  0.301
```

| dtype | fraction bits | significant digits |
|---|---|---|
| float16 | 10 | 10 x 0.301 = ~3 digits |
| float32 | 23 | 23 x 0.301 = ~7 digits |
| float64 | 52 | 52 x 0.301 = ~15 digits |

---

## Practical example: saving imaging data as float16

**Baseline TIFF** (raw fluorescence intensity, values ~20000):
- Exponent = 14, step = 16
- float16 can only store: ..., 19984, 20000, 20016, ...
- Decimal part is always 0 -> looks like "all zeros" in ImageJ
- **Use float32 instead** ✗ float16

**dF/F0 TIFF** (percentage values, ~0.5% to 50%):
- Exponent = 2 to 5, step = 0.004 to 0.03
- float16 stores decimal places just fine
- **float16 is fine** ✓ saves 50% disk space vs float32

---

## Why "double precision" = float64

"Double" means double the bits of float32:
```
float32  ->  single precision  (32 bits)
float64  ->  double precision  (64 bits = 32 x 2)
```
