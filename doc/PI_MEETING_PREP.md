# PI Meeting Preparation - ACh Imaging Analysis

**Status:** Data collected, refining analysis pipeline for PI discussion
**Last Updated:** 2026-02-12

---

## 📊 Dataset Summary

**What You Have:**
- Complete year of ACh imaging experiments (Jan 2025 - Jan 2026)
- Working analysis pipeline (spike detection + spatial categorization)
- Database with all results and metadata
- Complete plots for all experiments (spatial + region analysis)

**Key Observations:**
- Hotspots show substantial spatial extent
- Temporal persistence exceeds spike duration
- Gaussian spatial filtering required to detect hotspots
- Not every spike produces detectable release

---

## 🎯 Three Research Questions for Discussion

### 1. Temporal Dynamics: Information vs State Coding ⏱️

**Observation:** Hotspots last >200 ms while spikes are ~1 ms

**Key Question:**
> Do hotspots encode spike timing (information transfer) or create permissive chemical states (neuromodulation)?

**Approach:**
- **Pulse train analysis:** Does 5 spikes @ 50Hz create:
  - Summation (growing hotspot) → Information coding
  - Saturation (same size) → State creation / binary signal
  - Larger area (recruitment) → Network engagement

**Possible mechanisms:**
- Sensor kinetics (GACh3.0 slow unbinding?)
- Diffusion dynamics (ACh spreading)
- Receptor amplification (nAChR-mediated effects)
- Clearance rates (AChE activity)

**Can analyze NOW:** ✅ Spike train experiments in existing data

---

### 2. Spatial Targeting: What Determines Hotspot Localization? 🎯

**Observation:** Volume transmission is confirmed
- Spatial extent: 250-300 μm >> synaptic cleft (100 nm)
- Temporal persistence: >200 ms >> spike duration (~1 ms)

**Key Question:**
> What determines where ACh accumulates—receptor targeting or extracellular space (ECS) properties?

**Two Hypotheses:**

**Hypothesis 1: Receptor-Targeted Volume Transmission** 🎯
- ACh release is **directed toward** M1-receptor-expressing regions
- Hotspots preferentially overlap with M1+ postsynaptic targets
- **Test:** Overlay hotspot maps with M1 receptor staining
- **Prediction:** Hotspot boundaries align with M1+ regions

**Hypothesis 2: ECS-Shaped Random Diffusion** 🌊
- ACh spreads **randomly** from release site
- Hotspot localization is determined by **AChE distribution** in the ECS
- Patch regions (low AChE) → ACh accumulates → bright fluorescence
- Matrix regions (high AChE) → ACh rapidly cleared → dim/no signal
- **Test:** Overlay hotspot maps with patch/matrix compartments (AChE staining)
- **Prediction:** Bright hotspots in patch (low AChE), dim in matrix (high AChE)

**Approach:**
- Identify anatomical compartments (patch/matrix markers)
- Map M1 receptor distribution
- Overlay hotspot distributions with both markers
- Distinguish receptor-targeting vs ECS-shaping mechanisms

**Requires NEW experiments:** Anatomical/receptor staining + image registration

---

### 3. Validating Optimal Spatial Filtering Scale 🔬

**Observation:** Different Gaussian σ values reveal different hotspot sizes and intensities (see `output/gaussian/*.png`)

**Key Question:**
> What is the biologically valid spatial filtering scale that matches true ACh spread?

**Why This Matters:**
- Too small σ → noise dominates, overestimate hotspot count
- Too large σ → signals merge/flatten, underestimate hotspot count, lose spatial detail
- Need biological validation to ensure measurements reflect true ACh dynamics, not filtering artifacts

**Validation Approaches:**

| Method | Evidence Needed | Can Do Now? |
|--------|----------------|-------------|
| **SNR optimization** | Existing data + analysis | ✅ Yes (1-2 days) |
| **Consistency analysis** | Compare variability across experiments | ✅ Yes (2-3 days) |
| **Literature comparison** | ACh diffusion distance in striatum | ✅ Yes (lit review) |
| **Anatomical validation** | ChI terminal arbor size | ⚠️ If data exists |
| **Receptor correlation** | M1/M4 receptor distribution | ❌ Needs new experiments |

**Priority:** SNR optimization + literature comparison provides objective, data-driven validation

**Can analyze NOW:** ✅ Quantitative σ optimization with existing data

---

## 📈 Analyses You Can Run Immediately

### Priority 1: Pulse Train Summation Analysis
**Goal:** Test if hotspots accumulate with multiple spikes

**Method:**
1. Identify experiments with spike trains (ISI < 200ms)
2. Extract imaging segments around spike trains
3. Measure hotspot intensity/area after each spike
4. Compare: 1 spike vs 2 spikes vs 5 spikes

**Expected time:** 1-2 days

---

### Priority 2: Optimal Gaussian σ Validation
**Goal:** Determine biologically valid spatial filtering scale

**Method:**
1. Test different Gaussian σ values (3, 4, 5, 6, 8, 12, 16 pixels)
2. For each σ, measure:
   - Signal-to-noise ratio (SNR) in hotspot vs background
   - Hotspot size distribution consistency across experiments
   - Number of detected hotspots
3. Plot SNR vs σ to find optimal filtering scale
4. Compare optimal hotspot size to literature values for ACh diffusion distance

**Expected time:** 2-3 days

**This validates your current analysis pipeline and ensures measurements reflect true biology!**

---

### Priority 3: Temporal Dynamics Characterization
**Goal:** Measure hotspot rise/decay kinetics

**Method:**
1. Extract time courses of individual hotspots
2. Fit exponential rise/decay functions
3. Compare time constants across experiments
4. Relate to spike patterns

**Expected time:** 2-3 days

---

## 🗂️ Files & Resources

### Data Files
- Database: `results/results.db` (all experimental results and metadata)
- Individual experiment folders in `results/{date}/` (organized by date)
- Gaussian filtering comparison images in `output/gaussian/`

### Available Scripts
- `im_preprocess.py` - Image preprocessing pipeline
- `im_dynamics.py` - Temporal dynamics analysis

### Gaussian σ Comparison Images
See `output/gaussian/*.png` for visual comparison of different filtering scales:
- σ=3: Noisy, fragmented
- σ=6: Clean, well-defined (current default)
- σ=16: Over-smoothed, loss of spatial detail

**Use these images to demonstrate the importance of σ validation (Question 3)**

---

## 💬 Discussion Framework for PI

### Opening Statement
> "I've completed a year of ACh imaging analysis showing release from single cholinergic interneurons. The data reveals substantial spatial extent and temporal persistence of release events. This raises three questions about ACh volume transmission in the striatum."

### Present Three Questions
1. **Temporal:** Information coding vs state creation?
2. **Spatial:** Random diffusion vs compartment targeting?
3. **Methodology:** What is the biologically valid filtering scale?

### Identify Analysis Priority
- Questions 1 & 3: Can analyze with existing data (1-2 weeks)
- Question 2: Requires new experiments (anatomical staining)

### Ask for Direction
**Key questions for PI:**
- [ ] Which direction is highest priority?
- [ ] Should I focus on temporal dynamics or spatial coherence first?
- [ ] Is compartment-targeting worth pursuing now, or save for follow-up?
- [ ] Do we need GACh3.0 vs iAChSnFR sensor comparison?
- [ ] What timeline for complete analysis? (suggest 2-3 weeks)

---

## 📋 Next Steps (Based on PI Input)

### If PI Says: "Focus on temporal dynamics"
- [ ] Implement pulse train analysis
- [ ] Characterize hotspot kinetics
- [ ] Compare single spike vs spike train responses
- [ ] Test sensor kinetics hypothesis (if GACh3.0 vs iAChSnFR data available)

### If PI Says: "Focus on methodology / filtering validation"
- [ ] Run SNR optimization across different σ values
- [ ] Validate optimal Gaussian filtering parameters
- [ ] Compare hotspot sizes to literature ACh diffusion distances
- [ ] Assess consistency of measurements across experiments

### If PI Says: "Need sensor comparison"
- [ ] Identify GACh3.0 vs iAChSnFR experiments
- [ ] Compare temporal kinetics between sensors
- [ ] Compare spatial detection sensitivity
- [ ] Statistical comparison of key metrics

### If PI Says: "Get anatomical data"
- [ ] Plan staining experiments (patch/matrix markers, M1)
- [ ] Acquire images with anatomical markers
- [ ] Develop image registration pipeline
- [ ] Overlay hotspot maps with compartments

---

## 🔬 Literature to Review

**Before deeper analysis, check:**
- [ ] GACh3.0 and iAChSnFR sensor kinetics papers (temporal resolution)
- [ ] ChI terminal structure and axon arbor size (spatial scale)
- [ ] **ACh diffusion distance in striatum** (validates optimal σ) ← Priority for Question 3
- [ ] Striatal patch/matrix organization
- [ ] ACh volume transmission and clearance (AChE activity)
- [ ] Nicotinic receptor signaling timescales

**Key papers to find:** (2-3 foundational papers in each area)

---

## ✅ What You've Accomplished

- [x] Complete year of ACh imaging data collected (Jan 2025 - Jan 2026)
- [x] Database with experimental metadata and results (`results/results.db`)
- [x] Gaussian filtering parameter exploration (`output/gaussian/`)
- [x] Image preprocessing pipeline (`im_preprocess.py`)
- [x] Temporal dynamics analysis tools (`im_dynamics.py`)
- [x] Clear mechanistic questions identified

**Next:** Refine analysis pipeline to extract quantitative metrics for the three research questions

---

## 📞 Quick Reference

### Data Overview
- Complete year of experiments (Jan 2025 - Jan 2026)
- Multiple slices and cells analyzed
- All data stored in `results/results.db`
- Gaussian filtering examples in `output/gaussian/`

**Note:** Analysis pipeline needs to be refined to extract specific statistics (spike counts, hotspot areas, temporal dynamics) for PI presentation.

---

**Remember:** This meeting is about aligning on direction and priority, not defending completed work. You're asking for scientific guidance, which is exactly what PI meetings are for! 🧠✨
