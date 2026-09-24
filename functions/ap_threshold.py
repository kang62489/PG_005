"""
AP threshold of one spike: onset of the upstroke, where dV/dt first reaches DVDT_THRESH.

  Step 1. Peak     : Vm maximum of the trace
  Step 2. Upstroke : fastest-rising sample within UPSTROKE_SEARCH_MS before the peak
  Step 3. Onset    : walk back from there while dV/dt >= DVDT_THRESH -> threshold sample

Also: pre-spike baseline Vm, so thresholds can be reported relative to it (cancels any Vm offset).

Example:
    >>> hit = find_ap_threshold(time_ms, vm)       # (t_ms, v_mV) or None
    >>> base = baseline_vm(time_ms - t_peak, vm)   # mean Vm, -20 to -5 ms before the peak
"""

## Modules
# Third-party imports
import numpy as np

DVDT_THRESH = 20.0              # mV/ms: standard AP-threshold rate criterion (tunable)
UPSTROKE_SEARCH_MS = 5.0        # ms before the peak searched for the upstroke (skips earlier spikes)
BASELINE_WINDOW_MS = (-20.0, -5.0)  # ms from the peak: pre-spike baseline window


def baseline_vm(t_rel_ms: np.ndarray, vm: np.ndarray) -> float | None:
    """Mean Vm in BASELINE_WINDOW_MS (t_rel_ms = time from the spike peak), or None if no samples."""
    in_win = (t_rel_ms >= BASELINE_WINDOW_MS[0]) & (t_rel_ms <= BASELINE_WINDOW_MS[1])
    return float(vm[in_win].mean()) if in_win.any() else None


def find_ap_threshold(
    time_ms: np.ndarray,
    vm: np.ndarray,
    dvdt_thresh: float = DVDT_THRESH,
) -> tuple[float, float] | None:
    """(time_ms, Vm_mV) of the AP threshold, or None when no upstroke reaches dvdt_thresh."""
    # Step 1. peak
    peak_idx = int(np.argmax(vm))
    start_idx = int(np.searchsorted(time_ms, time_ms[peak_idx] - UPSTROKE_SEARCH_MS))
    if peak_idx - start_idx < 1:
        return None

    # Step 2. fastest rise before the peak; dvdt[i] is the slope from sample i to i + 1
    t_win = time_ms[start_idx : peak_idx + 1]
    v_win = vm[start_idx : peak_idx + 1]
    dvdt = np.diff(v_win) / np.diff(t_win)
    max_rise = int(np.argmax(dvdt))
    if dvdt[max_rise] < dvdt_thresh:
        return None

    # Step 3. onset of that supra-threshold run
    onset = max_rise
    while onset > 0 and dvdt[onset - 1] >= dvdt_thresh:
        onset -= 1
    return float(t_win[onset]), float(v_win[onset])
