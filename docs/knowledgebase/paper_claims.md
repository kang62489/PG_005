---
keywords: paper, claims, spontaneous, zones, coverage, synchrony, not in phase, single neuron, single spike, hotspot size, area comparison, locality, wave, flow, anisotropic, source, sink
files_referenced: spontaneous_analysis.py, classes/sp_zone_analyzer.py, ach_domain_analysis.py, classes/region_analyzer.py, area_comparison.py
related: zone_analysis_speedup_techniques.md
---

# 2026-09-27

## The three major claims of the project (confirmed by the user)

This is the shared reference for what the analysis pipeline is meant to prove.
Every new analysis or statistic should say which claim it supports.

---

## Story step 1 — Problem (final, 2026-10-05; no data, motivates all claims)

- **ACh compartment** = an area whose ACh level is distinct from its surroundings in both space and time.
- Striosome / matrix (found by AChE staining) are a **precedent only**; our compartments are not claimed to match them.
- Full wording and decisions: `docs/paper_discussion_2026-10-02.md`, section "Step 1: Problem (final, 2026-10-05)".

---

## Claim 1 — Spontaneous ACh forms local zones that cover the striatum, but not in phase

- **Pipeline:** `spontaneous_analysis.py` (10X only), output in `results/spontaneous/`
- **Measures:** zones per recording, zone area, event frequency, striatum coverage (needs `--stbd data/bd_20260922_000.json`)
- **Evidence so far (saion job 4734460, 105 recordings):**
  - 70 / 105 recordings have zones, 1,330 zones in total
  - median zone area ~102,700 µm², median frequency 0.11 Hz (period ~9 s), median coverage 0.56
  - 35 recordings with 0 zones are truly silent (log: "no hotspots above threshold"), clustered on
    2025_12_14 (16), 2025_11_27 (10), 2025_12_18 (7), 2024_10_11 (2, iAChSnFR)
- **Gap:** "not in phase" needs an event-timing (synchrony) measure between zones.
  Frequency spread alone does not show it.

---

## Claim 2 — One spike of one neuron releases a hotspot >= natural spontaneous hotspots

- **Meaning:** a local striatal area can be modulated by a single cholinergic interneuron.
- **Pipeline:** `ach_domain_analysis.py` (MED hotspot) + `area_comparison.py` (vs spontaneous zones)
- **Groups:**
  - A = estim-induced MED hotspots
  - B = spontaneous-spike MED hotspots
  - C = spontaneous zones not matched to the MED (nearest centroid >= 50 px)
- **Test:** Mann-Whitney U, two-sided, A vs C, B vs C, A vs B. Expectation: A, B > C.
  Also per recording: `pct_zones_smaller` = % of unmatched zones smaller than the MED.
- **Evidence so far (deigo job 45233657, 163 experiments):**
  - 10X detected: 45 / 92 (estim 30, spontaneous 15); median spike-frame hotspot 58,119 µm²
- **Status:** formal `area_comparison.py --db results/results.db` run pending.

---

## Claim 3 — Released ACh stays local instead of spreading as a wave

- **Pipeline:** `ach_domain_analysis.py` flow step (TV-L1 optical flow, striatum-masked), `flow_pairs` table, `*_FLOW.png` / `*_STREAMLINES.png`
- **Measures:** per frame pair, pattern = anisotropic vs source / sink; drift / spread (µm); DV / ML direction
- **Evidence so far (deigo job 45233657):**

  | OBJ | recordings | anisotropic | source / sink |
  |---|---|---|---|
  | 10X | 41 | 61 % | 39 % |
  | 40X | 9 | 81 % | 19 % |
  | 60X | 17 | 80 % | 20 % |

- **Origin:** old TODO "#1 wave vs hotspots" + "#5 locality argument".
