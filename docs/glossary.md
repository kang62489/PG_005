# Glossary

Terms used in the code and docs, after the 2026-10-08 rename (branch `major_refactor`, from the discussion with Jeff).

---

## 1. Renamed

| Old | New | Meaning |
|---|---|---|
| hotspot | **flash** | One connected blob of above-threshold pixels in **one frame** (after mask cleanup; fragments within 75 px merged; blobs > 80 % of the frame dropped). |
| compartment | **recur_zone** | A zone where flashes recur; the only zones counted in stats. |

### Spelling rules

| Old | New |
|---|---|
| `hotspot` / `hotspots` | `flash` / `flashes` |
| `Hotspot` / `HOTSPOT` | `Flash` / `FLASH` |
| `compartment` / `compartments` | `recur_zone` / `recur_zones` |
| `Compartment` / `COMPARTMENTS` | `Recur zone` (prose, sheet names) / `RECUR_ZONES` |

### Examples

| Old | New |
|---|---|
| `MAX_HOTSPOT_FRAC` | `MAX_FLASH_FRAC` |
| `spatiotemporally_connect_hotspots()` | `spatiotemporally_connect_flashes()` |
| `{stem}_HOTSPOT_MASK.tif` | `{stem}_FLASH_MASK.tif` |
| `hotspot_origin` (DB) | `flash_origin` |
| `spike_frame_hotspot_um2` (DB) | `spike_frame_flash_um2` |
| `detect_hotspot()`, `hotspot_area_um2` | `detect_flash()`, `flash_area_um2` |
| `compartment_id`, `n_compartments` | `recur_zone_id`, `n_recur_zones` |
| sheet "compartments" | sheet "recur_zones" |

---

## 2. Unchanged

| Term | Meaning |
|---|---|
| **unit** | Flashes linked over consecutive frames (circle rule). |
| **event** (`n_events`) | One run of consecutive active frames inside a recur_zone. |
| **zone** | Any mapped region (recur_zone or NR zone). |
| **NR zone** | Non-recurring zone; reference only, not in stats. |
| `SpontaneousZoneAnalyzer`, `sp_zone_analyzer.py`, `{stem}_ZONES.xlsx` | File and class names keep "zone". |

---

## 3. Not renamed

- Dated records (`.claude/plans/`, `docs/paper_discussion_2026-10-02.md`, history in `docs/continue_from_here.md`) keep the old terms.
