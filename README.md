# SWV Analysis UI

Interactive Streamlit app for batch SWV electrochemistry analysis.

## Setup

```bash
cd swv_app
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

If `streamlit` is installed globally on Windows instead of in the active virtualenv, you can also run:

```bash
py -m streamlit run app.py
```

The app opens automatically at http://localhost:8501

## Project layout

```
swv_app/
├── app.py              ← Streamlit UI (sidebar params, tabs, export)
├── requirements.txt
└── core/
    ├── io.py           ← File discovery, CSV loading, NaN filtering
    ├── processing.py   ← Smoothing, peak detection, baseline correction
    ├── analysis.py     ← Single-file analysis, partial failure traces, run_batch()
    └── plotting.py     ← All figure-returning plot functions
```

## UI tabs

| Tab | What it shows |
|-----|---------------|
| 🌈 Overlays | Colormapped raw / smoothed / corrected traces per channel |
| 📊 Metrics | Peak current, skew, wavelet energy vs scan — all channels combined |
| ⚠️ Failures | Failed trace plots + single-trace inspector |
| 🗂 Data Table | Filterable results table |
| 💾 Export | Download results.csv and a ZIP of all figures |

## Titration mode

If you enable `Treat vline intervals as titration steps` in the sidebar, the app adds an opt-in titration analysis layer on top of the normal scan-by-scan metrics.

- Each interval between consecutive vertical lines becomes one titration step.
- Plateau values are estimated per channel and per metric using the median of the middle portion of each step.
- Metric plots gain horizontal step plateaus, midpoint markers, and a smooth bridge through the step centers.
- An optional Langmuir-style fit is drawn from plateau value vs. titration step index.
- When the Langmuir fit is enabled, the app also exposes a Langmuir fit summary table / CSV with the fitted baseline, amplitude, saturation step, and apparent `Kd`.
- The Data Table and Export tabs expose step-level titration summaries only when this mode is enabled.

Because the current Langmuir x-axis is the titration step index rather than a physical concentration, the reported `Kd` is an apparent `Kd` in step-index units. If your steps map to known concentrations, you can convert or refit externally against that concentration axis.

The plateau estimator trims a configurable fraction from both edges of each step before taking the median, which helps suppress transition scans immediately after an addition event.

## Peak finding and baseline correction

For each SWV trace, the app follows this sequence:

1. Crop the raw trace to the selected voltage range.
2. Smooth the cropped current with a Savitzky-Golay filter.
3. Find the dominant peak on the smoothed trace.
4. Search for one bracketing minimum to the left of the peak and one to the right.
5. Draw a straight-line local baseline through those two minima.
6. Subtract that baseline from the smoothed trace.
7. Smooth the corrected trace again and re-detect the peak.
8. Report the corrected peak height and corrected peak voltage.

### 1. Cropping

Starting from raw voltage and current arrays:

```text
v_raw, i_raw
```

the app keeps only the points inside the crop window:

```text
v_min <= v_k <= v_max
```

which gives the cropped arrays:

```text
v = {v_k},  i = {i_k}
```

All peak finding and baseline correction are done on this cropped trace.

### 2. Smoothing

The current is smoothed with a Savitzky-Golay filter:

```text
i_smooth = SG(i)
```

Conceptually, the filter fits a low-order polynomial within a moving window. If the local polynomial is

```text
p(v) = a0 + a1*v + a2*v^2 + ... + am*v^m
```

then the smoothed value at the center of the window is:

```text
i_smooth(v_c) = p(v_c)
```

This reduces noise while preserving the peak shape better than a simple moving average.

### 3. Dominant peak detection

The app searches the smoothed trace for candidate peaks and keeps the dominant one. In practice this is the valid peak with the largest smoothed current:

```text
k_peak = argmax(i_smooth[k]) over valid detected peaks
```

If no peaks pass the prominence filters, the algorithm falls back to the global maximum:

```text
k_peak = argmax(i_smooth[k])
```

The peak voltage from this first pass is:

```text
v_peak = v[k_peak]
```

This first-pass peak is used to define where the baseline anchors should be searched.

### 4. Left and right minima search

Let the user-selected minima search window be `W = minima_search_window_V`. The algorithm defines:

```text
L = {k : v_peak - W <= v_k < v_peak}
R = {k : v_peak < v_k <= v_peak + W}
```

These are the allowed left-side and right-side search regions around the peak. The bracketing minima are then chosen as:

```text
k_L = argmin(i_smooth[k]) for k in L
k_R = argmin(i_smooth[k]) for k in R
```

The two anchor points are therefore:

```text
(v0, y0) = (v[k_L], i_smooth[k_L])
(v1, y1) = (v[k_R], i_smooth[k_R])
```

If either side has no points inside the requested voltage window, the code falls back to using all points on that side of the peak.

### 5. Local baseline from the two minima

The local baseline is the straight line through the two anchor minima. Its slope is:

```text
m = (y1 - y0) / (v1 - v0)
```

and the intercept form is:

```text
b = y0 - m*v0
```

so the baseline at any voltage `v` is:

```text
B(v) = m*v + b
```

or equivalently:

```text
B(v) = y0 + ((y1 - y0) / (v1 - v0)) * (v - v0)
```

This line represents the local background under the peak, approximated as linear between the two bracketing minima.

### 6. Baseline correction

The corrected current is calculated point-by-point by subtracting that baseline from the smoothed trace:

```text
I_corr(v) = I_smooth(v) - B(v)
```

or in index form:

```text
I_corr[k] = i_smooth[k] - B(v_k)
```

At the two anchor minima, the corrected signal is approximately zero:

```text
I_corr(v0) = 0
I_corr(v1) = 0
```

So this correction removes both:

- vertical offset
- local linear tilt

That is why the code refers to the step as a rotate/offset correction.

### 7. Final corrected peak measurement

After baseline subtraction, the corrected trace is smoothed again:

```text
I_corr_smooth = SG(I_corr)
```

The dominant peak is then re-detected on the corrected trace:

```text
k_peak,corr = argmax(I_corr_smooth[k]) over valid detected peaks
```

The final reported values are:

```text
Peak voltage  = v[k_peak,corr]
Peak current  = I_corr[k_peak,corr]
```

So the app uses the first-pass peak only to place the baseline anchors, but the final reported peak position and peak height come from the baseline-corrected trace.

If the SWV peak source is set to `Corrected + smoothed`, then the reported peak height instead uses:

```text
Peak current_selected = I_corr_smooth[k_peak,corr]
```

In that mode, the same selected trace basis is also used for the derived SWV metrics that depend on the final peak location.

### 8. Interpretation

If the measured signal is thought of as

```text
I(v) = s(v) + p(v)
```

where:

- `s(v)` is a slowly varying background or sloped baseline
- `p(v)` is the actual SWV peak

then the line through the left and right minima is used as a local estimate of `s(v)`:

```text
B(v) ~= s(v)
```

and the corrected trace becomes:

```text
I_corr(v) = I(v) - B(v) ~= p(v)
```

This works well when the local baseline is approximately linear near the peak. If the true baseline is strongly curved, some residual baseline shape may remain after correction.

### 9. Effect of `minima_search_window_V`

The parameter `minima_search_window_V` changes the allowed regions `L` and `R`:

- Smaller values force the minima to be closer to the peak, making the correction more local but also more sensitive to noise or shoulders.
- Larger values allow the minima to be farther from the peak, which can be more stable but may span a region where the true baseline is less linear.

## Background drift metric

The app also computes a simple peak-excluded background metric from the full raw trace. Let the crop window be:

```text
v_min <= v <= v_max
```

and let the full raw current be:

```text
I_raw = {I_k}
```

Define the outside-crop index set:

```text
O = {k : v_k < v_min or v_k > v_max}
```

Then the background RMS is:

```text
Background RMS = sqrt( mean( I_k^2 ) over k in O )
```

or equivalently:

```text
Background RMS = sqrt( (1 / |O|) * sum_{k in O} I_k^2 )
```

This metric is intentionally computed outside the SWV crop window so the peak-analysis region does not directly drive the background estimate.

## Background drift metrics

For each channel, the app uses the median background RMS of the first 3 valid scans as the reference:

```text
R_ref = median(R_1, R_2, R_3)
```

where `R_t` is the background RMS at scan `t` for that channel.

The normalized background level is:

```text
R_norm(t) = R_t / R_ref
```

The background drift fraction is:

```text
D(t) = R_norm(t) - 1
```

and the background drift percent shown in the UI is:

```text
Background drift (%) = 100 * D(t)
```

This RMS-based quantity is best treated as a diagnostic drift metric. Because RMS is always positive, it does not preserve the sign of an additive baseline shift, and it can increase either because the baseline moved or because the noise amplitude changed.

## Experimental additive background recentering

If you enable the experimental additive background recentering option in the SWV analysis sidebar, the app also computes the outside-crop median raw current for each scan:

```text
b(t) = median( I_k ) over k in O
```

Using the median of the first 3 valid scans in each channel as the reference background:

```text
b_ref = median(b_1, b_2, b_3)
```

the signed additive offset is:

```text
Delta_b(t) = b(t) - b_ref
```

The cropped raw SWV trace is then recentered before the usual baseline-correction workflow is rerun:

```text
I_recentered(V, t) = I_raw(V, t) - Delta_b(t)
```

The reported background-recentered peak is measured from that recentered trace after the standard SWV correction steps. This mode is opt-in and intended for comparison rather than as the default analysis path.

## Using core modules directly (no UI)

```python
from core import run_batch, plot_metric_vs_scan

results = run_batch(
    folders=["/path/to/data"],
    crop_range=(-0.61, -0.30),
    smooth_window=9,
    min_start_voltage=-0.7,
)

fig = plot_metric_vs_scan(results, metric="peak_current")
fig.savefig("peak_current.png")
```
