# SWV Analysis UI

Interactive Streamlit app for batch SWV/CV electrochemistry analysis and saved
Bayesian-optimization sessions.

## Setup

```bash
cd swv_app
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Or launch it by double-clicking the file for your operating system:

- Windows: `launch_app_windows.bat`
- macOS: `launch_app_mac.command`

The launcher creates a local `.venv` virtual environment when needed and
installs all packages from `requirements.txt` before starting the app.

On macOS, if the launcher is not executable after downloading it, run
`chmod +x launch_app_mac.command` once.

If `streamlit` is installed globally on Windows instead of in the active virtualenv, you can also run:

```bash
py -m streamlit run app.py
```

The app opens automatically at http://localhost:8501

## MATLAB SWV files

MATLAB files use the normal **SWV** analysis mode. Add folders containing
`.mat` files, then click **Convert MAT files for SWV**. Files are discovered
recursively and converted into native SWV CSVs in a `_mat_csv` folder, which
the app adds to the selected data folders automatically. Matching method files
are generated so the normal SWV frequency filters continue to work.

Matrices may be stored as either 2-by-N or N-by-2 arrays (voltage and current).
The converter prefers a variable named `data`, falling back to the first
numeric 2D matrix. Filenames such as `ch_5_scan_10_100hz_swv.mat` supply the
channel, scan number, and frequency. After conversion, all processing,
metrics, plots, filters, and exports are exactly the existing SWV pipeline.

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
| 💾 Export | Download CSVs, save a reusable experiment output bundle, and optionally export figures |

## Bayesian-optimization sessions

Choose **BO Session** under **Analysis mode**, then enter either an experiment
folder containing BO session subfolders or a specific `bo_session_<name>` folder
containing `bo_state.json`. If multiple sessions are found, choose the session
to inspect from the sidebar dropdown. The viewer reconstructs
both standard and paired-response BO sessions, including selectable metric
trends and history, per-channel scores, raw and corrected traces, 1D/2D/3D
surrogate views, measured-data landscapes, and candidate tables. Measured-data
landscapes plot non-Q metrics such as peak height directly against experimental
parameters without surrogate predictions. Raw/corrected traces are shown when
the recorded CSV files are still accessible beside or inside the session's
experiment folder. Corrected traces can use either the BO session's saved
parameters or editable native SWV processing settings in the sidebar.
Sessions with independently optimized channel groups are analyzed together by
default. Group-level trends remain separate, and observations are selected by
group plus iteration. The sidebar can optionally scope the analysis to one
group. Simulated sweep comparisons and loaded sweep-session trend plots can
color runs by the number of initial random/maximin points or by swept GP
falloff values, making it easier to compare convergence metrics versus
iteration. The simulation tab also supports compact multi-parameter sweeps:
enter separate lists for exploration, initial random/maximin points, GP falloff,
and repeats per hyperparameter point, then preview the generated Cartesian
product before running. Compact exports write only metadata, run summary, and
history CSV files, omitting generated SWV, analysis, surrogate, and tensor
artifacts. In **History & scores**, compact and sweep sessions expose a
hyperparameter response view that summarizes each run and plots the same
metrics available to the trend plot against two or three selected
hyperparameters, for example GP falloff versus exploration or a 3D
voxel hyperparameter response map.

The **PDF Export** tab builds a shareable, chronological report containing Q
definitions, objective and parameter evolution, channel and phase trends,
measured 1D/2D/3D landscapes, surrogate evolution for every saved artifact,
and all locally accessible raw and corrected SWV traces.

## Experiment output bundles

The Export tab can save a data-first bundle into an `outputs/` folder inside the selected data folder. If multiple data folders are selected, the app asks which one should receive the `outputs/` folder. These bundles are meant to be the input format for a future experiment-comparison app.

Example layout:

```text
MyExperiment/
  raw_csv_files...
  outputs/
    20260527_143200_MyExperiment/
      manifest.json
      swv_signal_processing_inputs.csv
      swv_results.csv
      swv_titration_steps.csv
      swv_langmuir_fit_summary.csv
```

The bundle does not need figures. Comparison plots should be reconstructed from the CSVs. `manifest.json` records the schema version, source folders, experiment name/notes, analysis mode, included files, and the signal-processing metadata needed to load the bundle reliably.

## Titration mode

If you enable `Treat vline intervals as titration steps` in the Scan Annotations panel, the app adds an opt-in titration analysis layer on top of the normal scan-by-scan metrics.

In the Metrics view, channels containing multiple SWV settings are automatically split into physical-channel/SWV-settings combinations, even when trace grouping is disabled elsewhere. Each method uses its own setting-local iteration axis and its own titration annotations. Interleaved methods remap source scan positions, while block-saved methods reuse the complete vline sequence on every local method axis. The detected-settings table matches frequency, amplitude, and step size to nearby saved BO observations and reports maximize/minimize only from the matched BO session configuration; missing or conflicting matches are shown as unresolved rather than inferred from the measured response. Choose **Individual channels per method** to create a separate metric plot for every combination; the same view also keeps plateau and Langmuir plots separated by method. In **Combined** view, use **Channels to combine** to select exactly which channels or displayed channel/method groups are overlaid; the subset also applies to combined plateau and Langmuir plots and the figures ZIP. When more than one peak-current trace is combined, each trace is offset by its selected anchor-buffer plateau and shown on one shared current-change axis: buffer is `0`, signal-on rises positive, and signal-off falls negative. Single-channel plots retain their original numerical current scale.

Use **All plot settings** at the top of the SWV Metrics view to change canvas size, text, lines, frame, markers, legend/grid visibility, and shared text overrides for every Metrics plot at once. Applying the form clears conflicting values previously saved in individual plot-settings popups.

The SWV **Overlays** view inherits the Metrics-wide visual-style defaults (including lines, frame, markers, margins, legend, and grid) while retaining its own axis labels. Its canvas keeps the shared Metrics height and uses 67% of the shared width without post-render bounding-box scaling. Overlay typography is compensated by the inverse width factor so titles, axis labels, ticks, legends, and colorbar text retain the same displayed proportions as Metrics plots on the narrower canvas. Overlay titles identify the settings-derived method and physical channel without assuming that the first two encountered methods are optimized and manual. The colorbar is labeled **SWV Measurement Number**.

When Langmuir fitting is active, **All Langmuir plot settings** provides the same shared controls scoped only to the fit figures. Langmuir figures are organized by physical channel, with that channel's settings-derived method fits overlaid for direct comparison.

### Vline annotation format

Write one vline per line:

```text
scan,label
```

The scan value is the x-axis position. The label is what appears on the marker. For titration/Kd fitting, start the label with the concentration for the step that begins at that vline:

```text
0, buffer
20, buffer
40, 10 uM
60, 20 uM
80, 40 uM
100, 80 uM
120, 160 uM
140, 320 uM
160, 640 uM
180, 1.28 mM
200, 2.56 mM
220, end
```

In this example, `buffer` is treated as `0` ligand, the `40` to `60` interval is fit as `10 uM`, the `60` to `80` interval is fit as `20 uM`, and so on. The last vline closes the final interval; its label can simply be `end`.

- Each interval between consecutive vertical lines becomes one titration step.
- Plateau values are estimated per channel and per metric using the median of the middle portion of each step.
- Metric plots gain horizontal step plateaus, midpoint markers, and a smooth bridge through the step centers.
- Optional Langmuir-style fits are drawn for both the selected SWV peak-height metric and wavelet energy versus vline-derived concentration when concentrations are present. Each metric receives its own fitted baseline, amplitude, `Kd`, buffer-noise estimate, and LOD.
- When both titration-step analysis and Langmuir fitting are enabled, the Metrics view and its figure exports show only metrics that produce at least one successful physical Langmuir fit under the current concentration and buffer selections. Unsupported diagnostics and supported metrics whose fit failed are hidden.
- Titration analysis automatically separates multiple SWV methods by their complete settings, with independently remapped intervals; explicit SWV-settings and modulo display groups use the same behavior. Interleaved methods are therefore fit independently rather than chronologically combined.
- Put the concentration at the start of each vline label, optionally followed by a note, for example `40,10 uM` or `40,10 uM, target added`.
- Each titration interval uses the concentration from its left vline marker.
- When the Langmuir fit is enabled, its default plot contains only the fitted curve, measured plateau points, and their error bars. Error-bar caps extend well beyond the marker width so small uncertainties remain visible even when their vertical span is hidden by a point. Peak-current Langmuir plots default to **Peak Height (uA)** on the y-axis. Enable **Show Kd and fit details on Langmuir plots**, **Show LOD on plots**, or **Show ULOQ on plots** to add those overlays. The app always exposes a fit summary table / CSV with the fitted baseline, amplitude, saturation step, saturation concentration, `Kd`, limit of detection (LOD), and upper limit of quantification (ULOQ). LOD uses `3 ×` the within-buffer standard deviation divided by the fitted Langmuir slope at zero concentration. ULOQ is projected from the fitted curve at the concentration where the remaining response to the Langmuir asymptote equals `3 ×` the median within-plateau SD of the highest half of the selected target concentrations: `Kd × (|amplitude| / (3σ_high) − 1)`. Selected-buffer noise is used only as a fallback when target plateau noise is unavailable. Extrapolated ULOQs are explicitly labeled, and the projected portion of the Langmuir curve is dashed when ULOQ display is enabled.
- Each selected concentration reports plateau SNR as `|plateau − fixed B| / median(selected-buffer SD)`. A per-SWV accuracy table also back-calculates concentration by inverting the fitted Langmuir response and reports absolute, signed-percent, absolute-percent, and log10 concentration errors for peak-current and wavelet-energy fits. These values are calibration residuals unless evaluated on an independently held-out run.
- The SWV Metrics page defaults to displaying only the selected Peak current metric. Wavelet energy and the other available metrics remain opt-in through **Metrics to display**; that selection also controls Langmuir, SNR, and concentration-diagnostic plots throughout the page. Every entry in **Channels to combine** receives one stable response color from the complete option list. Displayed SWV Method 1 traces always use one fixed blue shade and Method 2 traces always use a second fixed blue shade across every physical channel; other groups retain the response-direction palette. That exact channel-to-color mapping is reused in combined and individual metrics, plateau plots, Langmuir fits, SNR, concentration diagnostics, drift plots, and figure/PDF exports; selecting a subset or changing diagnostic grouping never reassigns shades. Metric y-axis labels, ticks, and spines remain black regardless of trace color. Direction comes from the Langmuir amplitude when available and otherwise from the median selected target-minus-preceding-buffer plateau change, so Langmuir plotting is not required. SWV methods use the same marker and line style and are differentiated by their shades. Combined mixed-direction iteration plots use one shared buffer-relative current axis and a zero reference line rather than separate y-axes.
- Combined wavelet-energy plots use the same buffer-relative translation: anchor-buffer energy is `0`, signal-on change is positive, and signal-off change is negative on one shared axis.
- Titration plateau plots mirror their corresponding iteration plots: combined peak-current and wavelet-energy plateaus use the same anchor-buffer translation and zero reference, while every plateau metric retains its assigned channel/method colors. Plateau axes for other metrics remain unchanged.
- Other metric-versus-iteration plots reuse the same channel response colors, but their values, scale, limits, and ordinary single y-axis are left unchanged.
- Concentration accuracy also reports RMSE in the selected concentration unit, calculated as `sqrt(mean((predicted - known)^2))` over finite predictions. It appears in the Metrics summary, the Data Table caption, and each predicted-versus-known plot; lower values are better and zero indicates perfect agreement.
- Use **SNR and concentration-accuracy plot grouping** to render both diagnostic plot families **Per channel** (overlaying only that physical channel's fitted SWV-setting groups), **Per SWV group** (one graph per channel/method combination), or **All groups** (the previous combined view).
- Each diagnostic group also receives a concentration-versus-measurement-number plot. Individual target and buffer SWV responses are converted to concentration by inverting that group's fitted Langmuir curve, then plotted against each method's local SWV measurement sequence. This avoids doubling the displayed measurement number when two methods are interleaved on the source axis. The x-axis is labeled **SWV Measurement Number**. Titration boundaries are remapped to and overlaid on that same method-local coordinate system as labeled vertical dashed lines, using the same annotation size as the peak-current-versus-measurement plots. This plot uses the uncensored inverse-Langmuir estimate rather than flooring predictions at LOD, so positive values below LOD and negative apparent concentrations remain visible; table values and the other concentration diagnostic retain their existing LOD reporting. Its y coordinates use doubling levels internally: zero concentration is `0`, the lowest selected nonzero dose is `1`, and `y = 1 + log2(C / Cmin)` above that dose, so every concentration doubling occupies exactly one vertical interval. Concentrations at or below the lowest dose use the monotonic linear mapping `y = C / Cmin`, which extends through zero for negative estimates. The visible y-axis defaults to **Predicted Concentration (uM)** with ticks `0` and the original selected concentration values—transformed level numbers are not shown. Uncertainty endpoints are transformed separately to retain asymmetric intervals. The default y-axis range extends 10% of the complete displayed-value span below its minimum and above its maximum, including visible uncertainty endpoints. Open that graph's **Plot settings**, enable **Manual y-axis limits**, and enter doubling-level limits to override its range independently; the downloaded image uses the same limits. The known titration is a dashed step reference drawn above the prediction data; this view does not add a ±20% band or prediction-summary subtitle. Its narrow, three-row legend sits inside the upper-left plotting area below the vertical-line labels. Both this plot and predicted-versus-known concentration use the fixed dark and light blue method colors; reference and acceptance-region colors remain unchanged.
- The Metrics view places **Titration SNR and concentration accuracy** directly below the Langmuir plots, including SNR-by-concentration and per-SWV tables, median absolute error, the percentage of predictions within ±20%, and direct CSV downloads. The same detailed rows remain available in the Data Table and Export views.
- Each fitted peak-current and wavelet metric also receives an SNR-versus-concentration plot and a **Predicted vs. Known** concentration plot for every invertible SWV. The SNR plot retains a linear concentration axis, overlays the response-derived fit `SNR(C) = |A|C / [σ(Kd + C)]`, marks the horizontal `3σ` cutoff, and places the vertical lower boundary at the exact fitted SNR = 3 intersection. The accuracy plot uses matched log–log concentration axes, which makes the multiplicative `y = 0.8x`, `y = x`, and `y = 1.2x` lines parallel. Its data axes are square, while its outer canvas and rendered height match the concentration-by-measurement plots. It includes visible in-range LOD/ULOQ boundaries, dotted ±20% error boundaries, and a shaded acceptance region. Separate color-coded upper-left annotations report **Optimized RMS Fold Error** and **Manual RMS Fold Error**, calculated as `10^sqrt(mean(log10(predicted / known)²))`; the lower-right legend contains only **Within ±20%** and **1:1**. Open **Plot settings** directly beneath an accuracy plot and use **20% acceptance-region alpha** to adjust that plot's shaded region from fully transparent to opaque; each plot has an independent setting and defaults to `0.10`. Both dimensions of the accuracy plot are determined by the lowest and highest selected concentrations actually measured, with a 5% logarithmic-span margin on each side so edge concentrations remain legible; predictions and projected limits outside this padded range are clipped and invisible boundary lines are omitted. Langmuir plots remain linear and show LOD as a dotted concentration line and ULOQ as a dash-dot line. Clear **Show LOD on plots** to hide LOD lines, labels, fit-note text, and the SNR = 3 cutoff shading; clear **Show ULOQ on plots** to hide ULOQ lines, annotations, and projected curve extensions. Both calculated values remain in tables and exports when hidden.
- For every Langmuir fit, the baseline `B` is fixed to the buffer plateau immediately preceding the earliest selected target concentration. Thus, selecting all targets uses the buffer before the first target, while starting at `80 uM` uses the buffer immediately before `80 uM`. If no such buffer exists, no `Kd` or LOD is reported for that channel/method.
- For alternating buffer/target experiments, choose **Immediately preceding buffer** as the titration baseline mode. Each target plateau is drift-corrected as `target plateau − most recent buffer plateau + anchor buffer plateau`, where the anchor is the buffer before the earliest selected target. Only amplitude and `Kd` are optimized. Without this mode, raw selected target plateaus are fitted, but `B` is still fixed to the same selection-dependent anchor buffer.
- Use **Concentrations included in titration statistics** to omit selected doses (for example, equilibration points at the beginning or unstable points at the end). Repeated buffers are listed chronologically as `buffer_1`, `buffer_2`, and so on. A deselected buffer may still correct drift for its immediately following target, but it cannot define the fixed `B` anchor or contribute noise to LOD. The selection applies to plateau tables, plots, Langmuir fitting, `Kd`, LOD, and exports. Titration plateau plots crop to the first and last selected interval, hide raw points from omitted intervals, and do not connect across internal omissions.
- Enable **Remove extreme titration outliers** to exclude isolated values independently within each channel/method and concentration interval. The raw-response filter uses a robust modified z-score cutoff of `5` (with a majority-consensus fallback when the median absolute deviation is zero). Because Langmuir inversion diverges near saturation, per-SWV predicted concentrations receive a second robust check in log-concentration space within each known dose. The filters apply consistently to metric and plateau plots, Langmuir fits, SNR, predicted-versus-known plots, tables, and figure exports.
- The Data Table and Export tabs expose step-level titration summaries only when this mode is enabled.

`Kd` is reported only when the fitted steps have numeric concentrations or `buffer` labels. Labels with units are converted into the selected Kd/report unit, and unitless numeric labels use that same unit.

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
