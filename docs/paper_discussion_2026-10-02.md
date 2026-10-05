# Paper discussion notes — 2026-10-02

Kang's self-talk on the paper story (PB_001), the agreed story flow, and checked discussion of each point.
Discussion only: **no code or results were changed.**

Sources used:
- Numbers: `D:\Programs\PG_005\results` (2.0σ, `MIN_OBJECT_UM2 = 900`, deigo 45504859; spontaneous saion 4734460), `RESULTS_GUIDE.md`.
- Paper drafts: `D:\Work\Vault_0\A4_Publications` (copy in `output/A4_Publications/`).
- Literature: `D:\Work\Vault_0\A5_Converted_Notes` (Brimblecombe & Cragg 2017, Matityahu 2023 + SI, Hamid 2021).

All "quick" numbers below are **descriptive back-of-envelope estimates**, not tested results.

---

## Kang's original self-talk (verbatim, 2026-10-02)

> Kept word for word as written, for hints and ideas to revisit. Sections 1–5 below summarise and discuss each point.

1. Why study this, the compartments in the striatum? Lack of clear anatomical sections on the basis of neural functions. Inspired by patches and matrix stained by AChE. It was found that the type of chemically defined area/compartments are actually related to certain behavior functions. ACh is the neurochemical, why not directly check it? We just have the kine of intensity based, genetically encoded tools designed for ACh.

2. So what can current results do? spontaneous ACh Zones. Temporally or Spatially linked hotspots. Found use global intensity historgram (n_bins 256) theshold at 2 sigma if I remember correctly. OK, why linking them? we want to see if there are groups of hotspots repetitively occur in the same place. If so maybe CINs are trying to "maintain" that area's ACh concentration -> could be a compartement (may like matrix, low [AChE] -> low hydrolysis -> high [ACh]). Footprints has highly correlated temporal curves, should be spatially very closed. hotspots with centroids are in proximity, well, really close in space. Left over, some hotspots which are occured in extremely low frequency or maybe like event-related. I want to check the zone sizes at 10X. The 40x and 60x's FOV just too small to see the whole hotspots/zones. It should be large as the range of a cholinergic interneurons' axon spread (Aosaki 1995). OK, the results of this part should let me claim there exists some large area, routinely replenished ACh zones which maybe potentially neurofunction related compartments <- cannot varified now, just propose this idea first. 😜. The first matrix and patches also just staining without any futher check.

3. How these zones are maintained? My guess -> by single Cholinergic interneuron. How am I going to proof? first, a release in normal situation is induced after an action potential (spike) activation (extracytosis). Therefore, I want to use single spikes as event, check if there is any hotspot occurs around the frame. However, right now from reliability analysis, due to the large variation of hotspot poistion and low reliablity, we can not guarantee the detected hotspots are really origin from patched neurons. Jeff suggested something like intrude rate esitmation, analyze and check the spike frequency, non-aligned analysis. Need figure out how to do. I also proposed roll back to original median all segments, then use the detected hotspots to check reliablity (I think this is probably much easier). Anyway, the hotspot sizes are the major things we want to look at to compare the sizes in spontaneous zones. If single neuron's single spike can manage that size of ACh zones, we may imply single neuron may able to manage a local striatal circuit (modulation). By the way, I think discussion is probably like further guessing to what we saw from the actual number of data, like reliablitiy 60% or 100% <- we may ask what happened? or are these number correct? and so on.

4. OK, now we should able to tell and summarize the previous experiemts to derive if a single cholinergic interneuron can create a compartment of not. Next, I also want to check some more properties of hotspots, that's why I fit the decay of area sizes and calculate the flow field. For decay of the area, maybe instead of showing time constants, I should use the fitted function to esitmate average lasting time (the decay time constant is not). Then we maybe compare the period of spontaneous zones to this to see if the matched. Therefore we may judge if really the compartments are really maintained. Lastly, the flow analysis, was originate papers that report the finding of ACh wavelike traveling. However, I still currently don't quite understand how Matityahu and Hamid proof this. Matityahu at least show some video of ACh pass though an ROI. Why I want to do this because I think this wavelike transmission may be the inter compartement communications. However, the current results seems to have pattern like, short source centered expandsion, then almost next frame sink like retraction. Lastly anisotropic fade away.

5. Considering all above and the core of article I think until point 3, we already complete the explanation, however, it will be a little bit not abundant if we have such less analysis. So I'm kind of struggling how the lasting time and flow analysis can be further discussed.

Follow-up replies (same day):
- On AChE: "I think you are right. 'In some sections, in the dorsolateral part of the head of the caudate nucleus, one or more zones appeared that were characterized by a cholinesterase content higher than that of the surrounding tissue.'"
- On the intrusion-rate plan sketch: "That will be great."
- Literature: converted markdowns in `D:\Work\Vault_0\A5_Converted_Notes`.

---

## Story flow (agreed 2026-10-03)

Marks: 🗣️ = Kang's self-talk (backbone), 💬 = Claude's additions. No statistics here — plain logic only.

1. **Problem** -- **final 2026-10-05**, see [Step 1 (final)](#step-1-problem-final-2026-10-05)
   - 🗣️ The striatum lacks anatomy that matches its functions; patch / matrix were first found by AChE staining.
   - 💬 AChE is a marker, not the ACh signal. Imaging ACh directly reveals compartments that are "hidden", i.e. invisible to staining.
2. **Spontaneous zones**
   - 🗣️ Hotspots recur at the same place and rise together.
   - 💬 Zones = places repeatedly supplied with ACh → candidate compartments. We *propose* them, we do not prove a function.
3. **Who supplies a zone?**
   - 🗣️ A single CIN? Treat single spikes as events; the key is the size comparison.
   - 💬 Weak point = location / chance → MED location + Jeff's intrusion rate. Evoked (current-driven) spikes are the cleaner single-cell case.
4. **What one CIN's area looks like**
   - 🗣️ Lasting time vs zone recurrence; flow = expand → retract → fade in place.
   - 💬 Broad + brief → release from many axon sites, not diffusion (§5.1).
   - Role of this block (Q3) is pending Kang's reading on how waves are defined.
5. **Meaning**
   - Hidden areas that are broad, brief and refreshed; one CIN is enough to make one.

**Term decision: "refreshed", not "maintained".**
- "Maintain" = held at a level. That is not what we saw (≈ 0.34 s on / ≈ 9 s off, §4).
- "Refreshed" = re-supplied in short pulses.

**Wave wordings (corrected).**
- Say: "Within the field and time window observed, the single-spike hotspot showed no directional spread." Slices cannot test for waves.
- "Waves need many CINs" is not our result → Discussion only, citing Matityahu's model.

---

## Step 1: Problem (final, 2026-10-05)

Story is now refined step by step (1 → 5); each step is fixed before moving on.

**Final wording:**

> The striatum lacks anatomical divisions that match its functions. A precedent shows that chemistry can reveal such divisions: AChE staining first exposed striosome and matrix, which were later found to serve distinct functions. ACh itself modulates striatal circuits, tuning SPN excitability and gating synaptic plasticity, so areas with different ACh levels would be modulated differently. Yet AChE is a static marker, and whether the ACh level itself differs across the striatum is unknown. We therefore imaged ACh directly and defined an **ACh compartment** as an area whose ACh level is distinct from its surroundings in both space and time. Such compartments need not match striosome or matrix; like them, they may carry their own functions.

➡️ *If we image ACh itself, do we see such areas?*

**Decisions:**
- **Definition:** ACh compartment = area whose ACh level is distinct from its surroundings in **both** space **and** time (not "or": zones need the same place *and* recurrence).
- **"ACh level"**, not "signal": the sensors (GACh3.0, iAChSnFR) are intensity-based and track concentration. "Level" reads as concentration without claiming absolute µM. The uncalibrated part goes in Methods only.
- **Striosome / matrix = precedent only** (chemistry → hidden functional units). Our compartments are **not** claimed to be striosome / matrix; their circuit details (limbic input, projection to DA neurons) are left out.
- **No "ACh drives DA release":** the direction is debated (Matityahu 2023: reciprocal CIN ↔ DA-axon coupling, DA can inhibit CINs). Keep only SPN excitability and plasticity (citations needed).
- **Static marker vs dynamic signal** is the novelty: AChE = enzyme snapshot in fixed tissue; imaging = the transmitter itself, live, over time.
- Literature gap kept: Brimblecombe & Cragg 2017, "unclear whether ACh itself" differs between striosome and matrix.

---

## 1. Why study compartments in the striatum

### Kang's view
- The striatum lacks clear anatomical sections that map onto neural function.
- Inspiration: striosome (patch) / matrix, first revealed by AChE staining. These chemically defined compartments later turned out to relate to behaviour.
- ACh is the neurochemical itself, so why not image it directly? Intensity-based genetically encoded ACh sensors now make that possible.

### Discussion
- The logic is short and strong: *the first compartments were found chemically → ACh is the chemical → image ACh directly.*
- **AChE direction (checked).** In the adult striatum, **striosomes are AChE-poor and the matrix is AChE-rich**:
  - Brimblecombe & Cragg 2017: the matrix "is enriched with … cholinergic markers including acetylcholine esterase (AChE) and choline acetyltransferase (ChAT)"; "AChE is found at higher levels in matrix".
  - Developmental twist (same review): the DA islands that become striosomes are AChE-**rich** in late embryonic / early postnatal life and switch to AChE-poor in adulthood.
  - Kang's quote from Graybiel & Ragsdale 1978 ("In some sections, in the dorsolateral part of the head of the caudate nucleus, one or more zones appeared that were characterized by a cholinesterase content higher than that of the surrounding tissue") describes an additional, less common observation; the main finding is AChE-poor striosomes. (Faull 1989 even proposed a third AChE-defined compartment.)
- **Superseded 2026-10-05:** the story no longer maps ACh zones onto striosome / matrix (see Step 1 final). The two points below stay as background only.
- **Kang's hypothesis, corrected direction:** low AChE → low hydrolysis → higher [ACh] would point to **striosomes**, not matrix. Brimblecombe & Cragg 2017 state exactly this idea and its alternative:
  > "Lower AChE levels in striosomes could be a surrogate marker for low ACh innervation density and low ACh levels, but alternatively, low AChE levels could limit ACh hydrolysis resulting in paradoxically higher ACh levels." They also note: "It is currently unclear whether ACh itself displays variable levels in striosomes versus matrix."
  → This is a direct, citable gap that ACh imaging can address.
- Bonus from the same review: ChIs "tend to occupy the boundary region between striosomes and matrix".

---

## 2. Spontaneous ACh zones

### Kang's view
- Hotspots linked temporally (trace correlation) or spatially (centroid proximity) → zones.
- Why link them: to find groups of hotspots that recur at the same place. If so, CINs may be "maintaining" that area's ACh level → a candidate compartment.
- Correlated footprints should also be spatially close; proximity-linked ones are very close in space. Leftovers are very-low-frequency or event-like hotspots.
- Zone size should be judged at 10X only (40X / 60X fields are too small). It should be comparable to a CIN's axon spread (Aosaki 1995).
- Claim: there are large, routinely replenished ACh zones that may be function-related compartments. This cannot be verified now; propose the idea first (patch / matrix also started as staining only).

### Discussion
- **Threshold detail (checked in code).** Threshold = background peak + **2σ** (`CROSSOVER_RATIO = 2.0`, `classes/sp_zone_analyzer.py:56`). The histogram has **512 bins** (`ZONE_HIST_BINS`), not 256, spanning the 0.1–99.9 percentiles (`ZONE_HIST_RANGE_PCT`, `functions/fit_hist.py:30-31`).
- **10X-only for size is right.** Zone median area 102,700 µm² → equivalent radius √(102,700 / π) ≈ **181 µm** (≈ 360 µm across).
- **Axon-arbor scale.** Matityahu 2023 (Discussion) uses a CIN axonal-arbor radius of "approximately 0.5 mm" (their ref. 102). Zones (r ≈ 180 µm) and 10X spike hotspots (r ≈ 160 µm) are well inside one arbor. One reading: a single spike activates only part of the arbor above threshold, or the 2σ threshold only captures its dense core.
- "Propose first, verify later" is a fair framing, with the patch / matrix history as precedent.

---

## 3. Who maintains the zones? Attribution to a single CIN

### Kang's view
- Guess: zones are maintained by single CINs.
- Release normally follows an action potential (exocytosis), so single spikes are used as events: is there a hotspot around the spike frame?
- Problem: hotspot position varies a lot and reliability is low, so detected hotspots can't be guaranteed to come from the patched neuron.
- Jeff's suggestion: an **intrusion-rate** estimate, i.e. check spike frequency and do a non-aligned analysis.
- Kang's proposal: roll back to the median of all segments, then use the detected hotspot to check reliability (probably easier).
- The main comparison is hotspot size vs spontaneous zone size. If one spike of one neuron can cover a zone, a single neuron may modulate a local striatal circuit.
- The Discussion is partly about questioning the actual numbers (e.g. why reliability is 60 % vs 100 %, and whether the numbers are right).

### Discussion

**How reliability is computed now** (`classes/spike_reliability.py:90-116`):
- Per raw segment: baseline threshold from that segment's own pre-spike frames → categorize spike and spike+1 → density-gated `detect_hotspot`.
- A segment is a "hit" if a hotspot passes the gate **anywhere in the frame**. The hit is **not** tied to the MED hotspot location.
- So current reliability = P(any hotspot in the field at spike / spike+1), which includes spontaneous zone events by chance.

**Quick chance estimate** (median spontaneous values: 15 zones, 0.11 Hz each, 20 Hz frames, window = 2 frames):

| Situation | Calculation | Chance hit |
|---|---|---|
| Only the zone at the MED spot | 0.11 Hz × 0.05 s × 2 | ≈ 1.1 % |
| Any zone in the 10X field | 15 × 0.11 Hz × 0.05 s × 2 | ≈ 16.5 % |
| Observed reliability, 10X | — | 66 % |
| Observed reliability, 40X / 60X | — | 13 % / 16 % |

- 10X: 66 % is far above both chance levels → supports a real spike-locked signal.
- 40X / 60X: reliability is in a range where chance could matter (though the smaller field also lowers chance). This is where the intrusion rate is most needed.
- Caveats: zone rates come from 10X spontaneous recordings, not the same recordings or objectives; the per-frame event probability ignores event duration (>1 frame) and the 2σ vs baseline-σ threshold difference.

**Evoked vs spontaneous hint.** Spontaneous-spike hotspots (132,694 µm², n = 18) are larger than evoked ones (74,383 µm², n = 26). One explanation: spontaneous spikes may coincide with other CINs firing, while current-evoked spikes are isolated. If so, evoked hotspots are the cleaner single-cell estimate. Good Discussion item; also relevant to the intrusion analysis.

**Literature support** (Matityahu 2023):
- "activation of a single CIN suffices to induce local striatal DA release (i.e., synchrony among several CINs is not required)".
- SI Fig. 2b: nine APs in one CIN at 0.2 Hz each triggered ACh release (GRAB-ACh3.0), with ≤ 4 % amplitude decrease from the 1st to the 2nd–9th response (n = 4 CINs).

---

## 4. Lasting time and flow

### Kang's view
- Fit the area decay; maybe report an "average lasting time" estimated from the fitted function instead of τ.
- Compare that with the period of spontaneous zones, to judge whether compartments are really "maintained".
- Flow analysis was inspired by papers reporting ACh wave-like travel (Matityahu; Hamid). Not yet clear how they showed it; Matityahu at least shows videos of ACh passing through an ROI.
- Motivation: wave-like transmission might be **inter-compartment communication**.
- Current result: short source-centred expansion → sink-like retraction almost in the next frame → anisotropic fade.

### Discussion — lasting time
- For a pure exponential A(t) = A₀·e^(−t/τ), the **mean lifetime equals τ** (∫A dt / A₀ = τ). So "average lasting time" from the fit is τ itself.
- A more intuitive number is the **time until the area falls below the detection limit**: t_end = τ · ln(A₀ / A_min).
  - 10X example: τ = 75 ms, A₀ = 80,900 µm², A_min = 900 µm² → 75 × ln(89.9) = 75 × 4.50 ≈ **337 ms** (≈ 7 frames).
  - Should be computed per recording from its own τ and A₀ (fit is from the peak frame to the end of the 21-frame stack; `lasting_time_ms` is NULL when R² < 0.8).
- **Comparison with the zone period:** ≈ 0.34 s on vs ≈ 9 s between events → a zone is ACh-high only ≈ **4 %** of the time. Zones are therefore **refreshed in short pulses**, not continuously maintained: *sparse in time, broad in space.* This reframes "maintain" rather than rejecting it.

### Discussion — how the wave papers measured waves

**Matityahu et al. 2023 (ACh waves, in vivo):**
- GRAB-ACh3.0 through a 3 mm cranial window in head-fixed mice (n = 3); also iAChSnFR via 1 mm window or GRIN lens (n = 6).
- Method: hand-drawn ROI → pixels binned into bands **perpendicular to the ML axis** → mean z-score per band per frame → **space–time plot** (1-D along ML). Diagonal streaks = waves.
- Wave location = band with maximal z each frame; its temporal derivative = instantaneous ML velocity (fluctuates within ±10 mm/s).
- **Bootstrapping:** location vector permuted in chunks of n = 1–20 frames (to respect slow indicator kinetics); compares run lengths between velocity sign reversals against the spurious maximum.
- Wave events: heuristic detector (runs ≥ 5 frames with consistent ML direction, plus intensity conditions); authors note the free parameters "were manually fine-tuned per movie".
- Results: GRAB-ACh3.0 waves every **5.2 ± 0.5 s**, duration **391 ± 9 ms**; iAChSnFR every **8.6 ± 1.2 s**, duration **537 ± 18 ms**; ~80 % lateral → medial.
- Mechanism (model): reciprocal CIN ↔ DA-axon coupling as a reaction–diffusion system; "Physical diffusion of DA and ACh from release sites is too short-ranged to produce the spread of activation observed in vivo". In some regimes, Turing "hills of activity" give "a dynamical parcellation or tiling [of the striatum] into distinct functional modules of high and low neuromodulatory activity".
- Slice data: electrically evoked DA release falls off over ~500 µm, halved by mecamylamine; CIN recruitment (ChAT-GCaMP6f) falls off over ~200 µm.

**Hamid et al. 2021 (*dopamine* waves, not ACh):**
- dLight and GCaMP6f in DA axons, widefield through a cranial window, in vivo.
- Method: optical flow (combined global–local, Lucas–Kanade + Horn–Schunck) between successive frames → pixel velocity field; **divergence** → source / sink locations; MATLAB `stream3` → flow trajectories; flow direction distribution (bimodal, ML axis); tdTomato control channel showed no ML bias.

**Relation to our pipeline:**
- Our flow analysis (TV-L1 → source / sink / anisotropic, streamlines) is methodologically parallel to Hamid's (optical flow → divergence → streamlines). Worth stating in Methods / Discussion.
- Their waves are population events over mm-scale windows; ours is one spike in a slice. Different questions, same toolbox.
- A Hamid-style null (their tdTomato channel) has a natural analogue here: run flow on random non-spike frames.

### Discussion — reading our flow sequence
- Observed: source (onset, spike−1 → spike+1) → sink (peaks spike+1 → +2) → anisotropic fade; drift 3.5 µm/frame (≈ 70 µm/s), no preferred direction.
- The "sink" phase may be **clearance from the edge inward**: the low-concentration rim drops below threshold first, so the region contracts. This is a clearance signature, not inward ACh movement.
- Compared with in vivo waves (±10 mm/s instantaneous), our drift is ~100× slower → within the field and time window observed (~200 ms after a single spike), no directional spread was seen.

### Striking match (worth checking)
- In vivo ACh waves: every 5.2–8.6 s, lasting 0.39–0.54 s (Matityahu 2023).
- Our 10X spontaneous zones: an event every ≈ 9 s (0.11 Hz); single-spike hotspot lasting ≈ 0.34 s (t_end above).
- The rate and duration are of the same order. Question: are zone events local views of wave-like events through a ~1 mm field, or independent local events? A direct check is the **zone event duration** (from each zone's active frames in `{rec}_ZONES.xlsx`), which we have not summarised yet.

---

## 5. How lasting time and flow can strengthen the story

### Kang's concern
- Up to point 3 the core argument is complete. With so few analyses the paper feels thin, so how can lasting time and flow be discussed further?

### Suggestions (ideas only)
1. **Size vs time → distributed release, not diffusion.**
   With an assumed tissue diffusion coefficient D* ≈ 0.4 µm²/ms (order of magnitude; needs a reference), ACh spreads √(4 × 0.4 × 75) ≈ **11 µm** in 75 ms (≈ 22 µm in 300 ms). The 10X hotspot radius is ≈ **160 µm** (√(80,900 / π)). One point source cannot explain this; the hotspot must come from **many release sites across the axon arbor at once**. This supports "single-CIN territory ≈ axon field" and matches Matityahu's statement that physical diffusion is too short-ranged. Probably the strongest use of τ.
2. **Pulsed compartments.** ≈ 0.34 s on / ≈ 9 s off gives compartments a temporal identity. It also sharpens the open point that CINs fire tonically at a few Hz while zones appear at 0.11 Hz.
3. **No directional spread from a single spike.** Within the field and time window observed, the single-spike hotspot showed no directional spread (slices cannot test for waves). That waves may need many CINs (and CIN ↔ DA-axon coupling) is Matityahu's model, not our result → Discussion only, with citation. This fits Kang's idea that waves are **communication between compartments**, not spread within one.
4. **Link to the compartment literature.** Matityahu's Turing-pattern "hills of activity" are a theoretical version of dynamic compartments; Brimblecombe & Cragg note it is unknown whether ACh differs between striosome and matrix.

**Overall view:** Claim 2 (attribution) carries the paper. Lasting time and flow are best presented as *what a single-cell compartment looks like*: broad (distributed release), brief (pulsed), cleared in place (no travel). They are supporting characterisation rather than separate claims.

---

## 6. Why "ACh in striosome vs matrix" could matter
1. **The compartments are functionally different.** Striosomes and matrix differ in inputs (striosomes get more limbic-cortex input), outputs (striosomal SPNs project directly to SNc dopamine neurons) and behaviour links. If ACh differs between them, so does cholinergic modulation of each circuit.
2. **ACh controls what the compartments do.** ACh drives / gates DA release via nAChRs on DA axons, tunes SPN excitability (M1 / M4), and gates corticostriatal plasticity. A compartment-specific ACh level means compartment-specific DA, excitability and learning.
3. **The DA analogue is already known.** Brimblecombe & Cragg 2017 describe an uneven "DA landscape" (more DA release in matrix than striosomes). Whether an "ACh landscape" exists is open.
4. **Marker ≠ signal.** AChE has been the compartment marker for ~50 years, yet nobody knows whether *ACh itself* follows it. Imaging the signal is the missing step; this is the direct link between "chemically defined compartments" (markers) and our "signal-defined zones".
5. **Two possible outcomes, both interesting.** If zones line up with striosome / matrix (e.g. by MOR co-staining later), the "hidden compartments" gain an anatomical anchor. If they don't, they are a new kind of compartment, defined only by the signal.
6. **CIN location.** ChIs sit at the striosome–matrix border, so which side their ACh mainly reaches is itself an open question.

---

## 7. Status of the claims

| Claim | Status |
|---|---|
| 1. Spontaneous ACh forms local zones that cover the striatum | kept. **"Not in phase" was dropped** (`PB_001_Story logic.md:90`: "Phase/timing between zones is *not* part of the story"). `RESULTS_GUIDE.md:13,32` and `docs/knowledgebase/paper_claims.md:16,25` still mention it → update later (user approval pending) |
| 2. One spike of one CIN produces a hotspot ≥ spontaneous zones | kept; attribution is the main weakness (§3) |
| 3. Hotspots are brief and spatially restricted | kept; looking for a stronger role in the story (§4–5) |

Evoked / spontaneous split of recordings with a detected hotspot (`has_region = 1`, `reliability_pct > 0`, n = 74;
`estim_induced` shown as **evoked**):

| OBJ | Evoked | Spontaneous | E : S |
|---|---|---|---|
| 10X | 26 | 18 | 1.44 : 1 |
| 40X | 4 | 4 | 1 : 1 |
| 60X | 15 | 7 | 2.14 : 1 |
| All | 45 | 29 | 1.55 : 1 (61 % evoked) |

Detection rate within each group (all `has_region = 1`): evoked 53 / 114 = 46 %, spontaneous 30 / 52 = 58 %.
