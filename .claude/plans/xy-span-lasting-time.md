# X/Y Span And Lasting Time Plan

## Summary

Add two per-recording measurements:

- X/Y span: spatial spread of the accepted event region on the max-area frame.
- Lasting time: temporal duration from critical frame until raw B+D% returns to baseline.

X/Y span is partly implemented already in the current worktree. Lasting time is not implemented yet.

## X/Y Span

- Keep measuring span on max_area_frame_idx, the larger raw B+D% frame among spike and spike+1.
- Use the DBSCAN-kept accepted mask, not raw B+D pixels:

    mask = label_frame >= 0

- Combine multiple clusters automatically by measuring one bbox over the whole mask.
- Store:

    max_area_x_span_px
    max_area_y_span_px
    max_area_x_span_um
    max_area_y_span_um

- Export only µm values to SQLite:

    max_area_x_span_um
    max_area_y_span_um

- Do not add span metrics to compute_region_stats().

Current status:

- RegionAnalyzer span calculation exists.
- ResultsExporter DB columns and migration exist.
- Console logging exists.
- region_sta title-only span display exists, with no image bbox overlay.
- Still needs full pipeline verification after a clean results.db run.

## Lasting Time

- Use the existing raw area_pct trace from compute_area_pct().
- Use the same baseline threshold style as pick_critical_frame():

    threshold = baseline.mean() + AREA_PCT_SIGMA_MULT * baseline.std()

- Define the event start as critical_frame_idx.
- Define the search start as the later of:

    critical_frame_idx
    np.argmax(area_pct[critical_frame_idx:]) + critical_frame_idx
    This avoids calling extinction before the event peak.

- Extinction frame = first frame after the search start where B+D% is below threshold for 2 consecutive frames.
- Lasting time:

    lasting_time_ms = (extinction_frame_idx - critical_frame_idx) * frame_duration_ms

- If no extinction frame is found, return/export None.

Add to RegionAnalyzer.get_results():

    extinction_threshold_pct
    extinction_frame_idx
    extinction_frame_offset

Add a method:

    get_lasting_time_ms(frame_duration_ms: float) -> float | None

Add DB columns:

    extinction_frame_offset REAL
    lasting_time_ms REAL

## Figure Display

- For x/y span:
    - Keep it text-only in second-row multirow titles.
    - No bbox or span overlay on image panels.
    - Show span only on the max-area frame panel.

- For lasting time:
    - Add top-row B+D% trace annotations:
        - horizontal threshold line
        - extinction-frame marker if found
        - optional light shading from critical frame to extinction frame

    - Do not add lasting-time text to every image panel.

## Test Plan

- Run Ruff on changed Python files.
- Unit-test span helper with:
    - empty mask

- Unit-test extinction logic with:
    - normal decay below threshold
    - one-frame dip ignored by 2-frame confirmation
    - no extinction found
    - spike+1 critical frame

- Full pipeline verification:
    - delete results/results.db
    - run ach_domain_analysis.py --ana_list data/ana_list_20260622_000.txt
    - confirm DB span/lasting-time columns exist and populate
    - inspect one saturated 60X and one 10X region_sta PNG

## Assumptions

- Span uses DBSCAN-kept pixels, not raw B+D.
- Multiple clusters count as one combined spread region for x/y span.
- Lasting time uses raw B+D%, because it follows the same temporal signal used to choose the critical frame.
- X/y span is stored per recording but excluded from aggregate stats.
