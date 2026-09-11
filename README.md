## Adora: Automatic Differentiation fOr Response functions using jAx

Adora is a research prototype for differentiable LTE scalar and polarized
radiative transfer. It synthesizes the Fe I 630.1/630.2 nm pair, computes JAX
response functions, and demonstrates full-depth, node-based, SciPy, Adam, and
PINN-style inversions.

### Installation and verification

Python 3.12 or newer is required. Runtime and test versions are pinned in
`pyproject.toml`:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest
```

The default numerical mode is JAX FP64. The 3-D PINN also supports an explicit
`--precision fp32` mode for consumer GPUs. Precision is configured once, before
physics tables are created, through `adora_precision.py`; setting
`ADORA_PRECISION=fp32` gives the same behavior to library imports and the other
scripts. Neural-network matrix products use full FP32 rather than TF32.

FP32 synthesis does not store wavelengths near 630 nm as absolute device
coordinates: their spacing would be too coarse at that magnitude. Absolute
wavelengths stay in host FP64 metadata and are transferred as offsets from a
fixed reference. Atomic and partition tables are cast to the selected real
dtype, Voigt calculations use the corresponding complex64/complex128 dtype,
and the Saha exponent is evaluated in a derivative-safe scaled form.

### Conventions and scope

- Atmospheres are ordered from the lower boundary toward the observer. Use
  `atmosphere_from_falc` rather than manually reversing FAL arrays or building
  cell widths.
- Scalar and polarized synthesis both use a Planck lower boundary by default.
- Intensities are in kW/(m2 nm sr), emissivities in kW/(m3 nm sr), opacities in
  m-1, and other quantities in SI units.
- Atomic populations and line synthesis currently support Fe I only. The
  Kurucz loader rejects other elements/stages instead of silently applying Fe I
  populations.
- Fe I/II/III partition functions are interpolated from Lightweaver's Kurucz
  table over 1,000--100,000 K.
- Polarized transfer currently assumes a vertical (`mu=1`) ray and uses the
  DELO-constant formal solver.
- The scalar solver treats each sample as a piecewise-constant cell and uses
  every `dz`. The polarized solver treats samples as depth points, uses
  `dz[i]` between points `i-1` and `i`, and therefore ignores `dz[0]`. Their
  zero-field spectra converge with refinement but are not bitwise-equivalent
  on a finite nonuniform atmosphere.

### Scripts

- `response_fn.py`: scalar synthesis and response functions.
- `vector_response_fn.py`: full-Stokes synthesis and response functions.
- `iterate.py`: bounded Levenberg--Marquardt inversion demo.
- `iterate_scipy.py`: SciPy trust-region inversion with a compiled Jacobian.
- `iterate_adam.py`: transformed-parameter first-order inversion demo.
- `nodes.py`: differentiable node reconstruction and polarized inversion.
- `create_2D_testset.py`: writes named `intensity` and `wavelength` arrays to
  `data/spectrum_2D.npz`.
- `PINN_example.py`: PINN training example using that dataset.
- `pinn_3d_inversion.py`: full-Stokes 3-D neural-field test-cube generation,
  fixed-reference log10 corrections, and spectropolarimetric inversion.
- `compare_inversion_gui.py`: linked 4 x 3 input/inversion spectra and
  horizontal-atmosphere viewer.
- `create_test_set.py`: configurable center-to-boundary full-Stokes contrast
  cube generation plus a decoder-compatible initialization checkpoint.
- `compare_pinn_precision.py`: isolated FP32/FP64 PINN accuracy and timing
  comparison with machine-readable spectra, gradient, convergence, and
  recovered-atmosphere metrics.
- `lte_pops.py`: generic LTE-population validation demo; it is not used by the
  Fe synthesis path.

Substantial demo work is protected by `if __name__ == "__main__"`, so importing
the modules does not start training, plotting, or data generation.

### Three-dimensional PINN spectropolarimetric inversion

`pinn_3d_inversion.py` implements an end-to-end synthetic experiment. By
default it:

1. Copies the 82-point FAL-C thermodynamic model over a `50 x 50` horizontal
   grid, adds a weak non-axis-aligned magnetic reference field and a small
   smooth 3-D perturbation, and synthesizes `(50, 50, 4, 201)` full-Stokes
   spectra.
2. Stores the exact FAL-C reference profiles and initializes one spatial MLP
   with a zero final layer. Its eight `tanh` outputs start at exactly zero at
   every `(x,y,z)`, so the atmosphere starts exactly at the reference without
   fitting a base network or running a pretraining optimizer.
3. Fits that shared spatial network with continuum-normalized, weighted
   Stokes residuals. Complete columns are shuffled into fixed-size batches;
   wavelength subsets are sampled uniformly, making each batch loss an
   unbiased estimate of the full-spectrum objective. Every Kurucz line center
   is included exactly in the saved wavelength grid. A fixed full-wavelength
   validation batch selects the best epoch, and fitting fails explicitly if no
   epoch improves on the initial reference or supplied warm start.

The network output `u` is dimensionless and bounded to `[-1, 1]`. For each
positive field `q` (temperature, electron density, total hydrogen density,
microturbulence, and unsigned magnetic strength), decoding at height `z` uses

```text
log10(q(x,y,z) / q_reference(z)) = spatial_scale[q] * u[q](x,y,z)
q(x,y,z) = q_reference(z) * 10**(spatial_scale[q] * u[q](x,y,z)).
```

With the default unit scales, `u=0` preserves the reference, `u=0.30103`
doubles it, and `u=-1` gives one tenth of it. This is a log ratio, not
`log10(q - q_reference)`. The raw final-layer activations pass through `tanh`
before they are interpreted as corrections. The prior penalizes these
normalized outputs equally across channels.

A log ratio is undefined for zero or signed reference quantities. LOS velocity
therefore uses `v = v_reference + 20000 m/s * scale[v] * u[v]`. Inclination uses
`gamma = gamma_reference + pi * scale[gamma] * u[gamma]`, reflected into
`[0, pi]` if it crosses a pole. Azimuth uses
`chi = chi_reference + pi/2 * scale[chi] * u[chi]`; the forward solver is
pi-periodic in azimuth. These definitions preserve zero velocity and zero
azimuth and have finite, nonzero derivatives at initialization. FAL-C has no
magnetic field prescription, so the configured nonzero magnetic seed remains
part of the reference; no pressure channel is fitted.

`--spatial-scale T,NE,NH,V,VT,B,GAMMA,CHI` sets eight positive scales (all one by
default). Positive-field scales are in dex, and the other three multiply the
velocity/angle units above. The normalized output stays in `[-1, 1]` even when
a scale is increased. A one-dex target exactly at an output bound needs a
scale greater than one to keep it reachable without saturating `tanh`.

At stored heights the reference is used exactly; at intermediate heights the
stored physical profiles are linearly interpolated. Every synthesis, loss,
validation evaluation, diagnostic figure and saved atmosphere uses the same
decoder and fixed reference. For datasets made by `create_test_set.py`, this
is the stored boundary reference, including any requested temperature anchor
or boundary velocity.

On a GPU or other accelerator, the independent columns inside each synthesis
batch are vectorized. Wavelengths are processed in sequential chunks, with the
wavelengths inside each chunk vectorized together; the default
`--wavelength-parallelism 48` matches the default training wavelength batch.
CPU execution retains the faster sequential column path. The PINN synthesis
also uses the nonsingular DELO specialization for positive atmospheres. This keeps the generic solver's expensive matrix
exponential fallback out of the batched differentiation graph; the generic
solver remains available for atmospheres that may contain singular opacity.

All ordered optimizer updates in each spectral epoch run in one compiled
`jax.lax.scan`. Loss histories
and epoch metrics are transferred to the host in bulk instead of synchronizing
the GPU after every update. The first call for a new combination of array
shapes and static options still includes JAX/XLA compilation time.

All Fe I records in the Kurucz file passed with `--kurucz` are summed by the
forward model. The default is the packaged 630.1/630.2 nm line list. Run the
complete default experiment with:

```bash
python pinn_3d_inversion.py
# or, after installation:
adora-pinn-3d
```

Select the consumer-GPU mode at process startup with:

```bash
adora-pinn-3d --precision fp32
# Equivalent for imports or another script:
ADORA_PRECISION=fp32 python pinn_3d_inversion.py
```

Generated datasets record `synthesis_precision`, fitted results record
`inversion_precision`, and checkpoints record their creation precision.
Checkpoint transform metadata and atomic fingerprints are precision-independent,
so an FP64 checkpoint or dataset can deliberately be loaded by an FP32 run.

Generation and inversion can be run separately:

```bash
python pinn_3d_inversion.py --generate-only --dataset data/pinn_3d_test_cube.npz
python pinn_3d_inversion.py --invert-only --dataset data/pinn_3d_test_cube.npz
```

### Creating a configurable center-to-boundary test set

`create_test_set.py` creates the stronger, exactly reproducible test atmosphere
used for center-to-boundary experiments. Its defaults are:

- `T(center) / T(boundary) = 1.10` at every FAL-C height;
- `ne(center) / ne(boundary) = 1.10` at every FAL-C height;
- LOS velocity from `-5 km/s` on every horizontal boundary to `+5 km/s`
  at the center; and
- signed LOS magnetic field from `-500 G` on every boundary to `+500 G`
  at the center.

By default, the normalized, height-independent contrast envelope is

```text
E(x,y) = [4 x (1-x) 4 y (1-y)]^profile_power.
```

Thus `E=0` wherever `x` or `y` is 0 or 1, while `E=1` at
`x=y=0.5`. Temperature and electron density interpolate through
`Q = Q_boundary [1 + (center_ratio - 1) E]`; velocity and signed LOS
field interpolate linearly between their endpoint values. Grid sizes must be
odd and at least three so the discrete cube contains the exact center.
Total hydrogen density and microturbulence remain equal to FAL-C everywhere;
the magnetic magnitude and azimuth are also spatially constant.

For a contrast requested at one photospheric height, pass
`--thermodynamic-height-km`. Temperature and electron density then use a
Gaussian vertical factor centered on the FAL-C depth point nearest that
height:

```text
E_thermo(x,y,z) = E(x,y) exp[-0.5 ((z-z_selected)/sigma_z)^2].
```

`--thermodynamic-sigma-km` sets `sigma_z`. The selected stored layer has a
factor of exactly one, so center/boundary ratios are exact there. Velocity and
magnetic contrasts continue to use the height-independent `E(x,y)`. An
optional `--temperature-boundary-k` smoothly rescales the reference
temperature around the selected layer so its boundary value is exactly the
requested temperature. The dataset records both the requested height and the
actual selected FAL-C grid height.

Magnetic strength itself is never negative. The generator holds the unsigned
strength `B` constant and stores an inclination satisfying
`gamma = arccos(B_LOS/B)`. The default `B=1000 G` therefore represents
`B_LOS=-500 G` with `gamma=120 degrees` at the boundary and `B_LOS=+500 G`
with `gamma=60 degrees` at the center. The azimuth remains constant.

Create the default FP32 test set on the first run with:

```bash
python create_test_set.py --precision fp32
# Equivalent after installation:
adora-create-test-set --precision fp32
```

This writes `data/pinn_contrast_cube.npz` and
`data/pinn_contrast_initial.npz`. Existing outputs are protected. To rerun the
same command and deliberately replace both files, add `--overwrite`:

```bash
python create_test_set.py --precision fp32 --overwrite
```

For a thermodynamic perturbation at the layer nearest 100 km with
`T_boundary=5000 K`, `T_center=6000 K`, a one-dex center increase in electron
density, and signed `B_LOS` from 0 G at the boundaries to 500 G centrally, use:

```bash
python create_test_set.py \
  --precision fp32 \
  --dataset data/pinn_blos_0_500_t5000_6000_ne1dex.npz \
  --checkpoint data/pinn_blos_0_500_t5000_6000_ne1dex_initial.npz \
  --thermodynamic-height-km 100 \
  --thermodynamic-sigma-km 100 \
  --temperature-boundary-k 5000 \
  --temperature-center-ratio 1.2 \
  --ne-center-ratio 10 \
  --boundary-velocity-km-s 0 \
  --center-velocity-km-s 0 \
  --boundary-b-los-gauss 0 \
  --center-b-los-gauss 500 \
  --field-strength-gauss 1000
```

Here one dex means a factor of ten. The `100 km` Gaussian width is an explicit
modeling choice; change `--thermodynamic-sigma-km` to make the vertical
perturbation narrower or broader. Add `--overwrite` only when intentionally
replacing an existing dataset and checkpoint.

Use `--thermodynamic-height-km` to confine the requested temperature and
electron-density contrast around that photospheric layer. Omitting it applies
the ratio at every height, including the hot upper atmosphere.

The magnetic endpoints above are the signed line-of-sight component, not the
unsigned magnitude: the latter remains 1000 G and the inclination supplies
the requested `B_LOS`. A magnetic-strength log ratio
cannot represent an unsigned field magnitude of exactly zero.

For a shell history that states every non-path numerical choice explicitly,
the equivalent default rerun is:

```bash
python create_test_set.py \
  --precision fp32 \
  --dataset data/pinn_contrast_cube.npz \
  --checkpoint data/pinn_contrast_initial.npz \
  --nx 51 --ny 51 --n-wave 201 \
  --wavelength-padding-nm 0.05 \
  --synthesis-batch-columns 16 \
  --wavelength-parallelism 48 \
  --profile-power 1.0 \
  --thermodynamic-sigma-km 100 \
  --temperature-center-ratio 1.10 \
  --ne-center-ratio 1.10 \
  --boundary-velocity-km-s -5.0 \
  --center-velocity-km-s 5.0 \
  --boundary-b-los-gauss -500.0 \
  --center-b-los-gauss 500.0 \
  --field-strength-gauss 1000.0 \
  --magnetic-azimuth-deg 0.0 \
  --spatial-hidden 96,96,96 \
  --spatial-scale-safety-factor 1.25 \
  --spatial-scale-padding 0.05 \
  --seed 0 \
  --overwrite
```

This uses the packaged default Kurucz file and automatic spatial-scale
selection. Supply `--kurucz` or `--spatial-scale` explicitly when either must
differ from those defaults.

The checkpoint contains an exactly zero-output spatial network, the stored
boundary reference profiles, and scales derived from the requested truth.
Use this checkpoint to preserve those reference profiles and correction
ranges when starting an inversion:

```bash
python pinn_3d_inversion.py --precision fp32 --invert-only \
  --dataset data/pinn_contrast_cube.npz \
  --checkpoint-in data/pinn_contrast_initial.npz \
  --checkpoint data/pinn_contrast_fitted.npz \
  --result data/pinn_contrast_inversion.npz
```

The initial checkpoint already returns the exact boundary atmosphere. There
is no pretraining step or `--skip-pretraining` flag. A fitted checkpoint can
be reused with `--checkpoint-in`; the dataset reference must match. Append
`--wandb --wandb-name contrast-default-fp32` when W&B tracking is configured.

#### Complete `create_test_set.py` argument reference

All numerical endpoint arguments are physical values, while
`--spatial-scale` uses dex for positive fields and the signed velocity/angle
units described above. Running `python create_test_set.py --help` prints the same options
and their defaults.

| Argument | Default | Detailed behavior |
|---|---:|---|
| `-h`, `--help` | — | Prints grouped command help and exits without generating anything. |
| `--precision {fp32,fp64}` | `fp64`, unless `ADORA_PRECISION` selects another mode | Selects all JAX real/complex dtypes before importing atomic tables or solvers. Direct CLI use is required to change an already imported process. FP32 is normally appropriate for an RTX 4090. |
| `--dataset PATH` | `data/pinn_contrast_cube.npz` | Destination for coordinates, reference/truth atmospheres, full-Stokes observations, contrast metadata, atomic fingerprint, required spatial scale, and selected spatial scale. Parent directories are created automatically. |
| `--checkpoint PATH` | `data/pinn_contrast_initial.npz` | Destination for the exact reference profiles, zero-output spatial MLP, architecture, precision metadata, and eight-channel correction scales. Use this exact file with inversion `--checkpoint-in`. It must differ from `--dataset`. |
| `--kurucz PATH` | packaged Fe I 630.1/630.2 nm line list | Atomic line list used to build the wavelength grid and synthesize observations. Inversion must use identical atomic data; a SHA-256 fingerprint enforces this. |
| `--seed INTEGER` | `0` | JAX PRNG seed for initial checkpoint weights. It does not alter the deterministic atmosphere, envelope, wavelength grid, or spectra for fixed numerical settings. |
| `--overwrite` | off | Permits replacement of both named output files. Without it, either existing output causes failure before expensive synthesis starts. Directories are never accepted as output files. |
| `--no-progress` | off | Suppresses the truth-cube synthesis progress bar; final output-path and scale messages are still printed. |
| `--nx ODD_INTEGER` | `51` | Number of normalized x samples from 0 through 1. It must be odd and at least 3 so index `(nx-1)/2` is exactly `x=0.5`. It contributes directly to the number of synthesized columns and archive size. |
| `--ny ODD_INTEGER` | `51` | Number of normalized y samples from 0 through 1, with the same odd-size and exact-center requirements as `--nx`. Total columns equal `nx*ny`. |
| `--n-wave INTEGER` | `201` | Number of vacuum-wavelength samples spanning the complete selected line list. Every line center is inserted exactly; too few samples for the endpoints and unique centers are rejected. |
| `--wavelength-padding-nm FLOAT` | `0.05` | Positive spectral margin in nm below the lowest and above the highest line center. Increasing it expands the continuum wings without changing `--n-wave`. |
| `--synthesis-batch-columns INTEGER` | `16` | Positive number of atmosphere columns synthesized concurrently. Larger values can improve GPU throughput but increase accelerator memory use. It changes batching only, not the generated atmosphere. |
| `--wavelength-parallelism INTEGER` | `48` | Positive maximum number of wavelengths evaluated concurrently within each synthesis chunk. Increase only while GPU memory permits; values above `--n-wave` add no parallel work. It changes execution strategy, not model values. |
| `--profile-power FLOAT` | `1.0` | Positive exponent applied to the center-to-boundary envelope. Values above 1 concentrate changes more tightly around the center; values between 0 and 1 broaden them. Endpoints remain exactly zero and one. The same envelope applies at every height. |
| `--thermodynamic-height-km FLOAT` | omitted | Opts into height-localized temperature and electron-density changes. The finite value is a requested geometric height in km; the nearest stored FAL-C layer is selected and recorded. At that selected layer the vertical factor is exactly one. When omitted, temperature and electron-density ratios retain their original height-independent behavior. This option does not localize velocity or magnetic changes. |
| `--thermodynamic-sigma-km FLOAT` | `100.0` | Positive Gaussian standard deviation in km for height-localized temperature and electron-density changes. It has no effect unless `--thermodynamic-height-km` is supplied. A smaller value confines the perturbation to fewer depth points; a larger value affects more of the atmosphere and can require larger correction scales. |
| `--temperature-boundary-k FLOAT` | omitted | Optional positive absolute boundary temperature in K at the selected thermodynamic layer. It requires `--thermodynamic-height-km`. The FAL-C reference temperature is smoothly rescaled with the same vertical Gaussian and is set exactly to this value at the selected layer. The center there is this value multiplied by `--temperature-center-ratio`. |
| `--temperature-center-ratio FLOAT` | `1.10` | Positive ratio `T_center/T_boundary`. Boundary temperature is the original FAL-C profile unless `--temperature-boundary-k` anchors it. Use `0.90` for a center 10% cooler than the boundary. Every generated value must be finite, positive, and reachable within the correction scales. |
| `--ne-center-ratio FLOAT` | `1.10` | Positive ratio `ne_center/ne_boundary`. Boundary electron density is FAL-C. A value of `10` gives a one-dex central increase. With height localization, this ratio is exact on the selected layer and smoothly returns toward one away from it. Nonphysical fields and unreachable correction ranges are rejected before synthesis. |
| `--boundary-velocity-km-s FLOAT` | `-5.0` | Finite LOS velocity assigned at all four horizontal boundaries and saved as the one-dimensional reference profile. Positive and negative values follow the solver's existing LOS sign convention. The default correction unit is `20 km/s` relative to this velocity. |
| `--center-velocity-km-s FLOAT` | `5.0` | Finite LOS velocity at `x=y=0.5`; intermediate pixels are a linear interpolation in the envelope. Together with the default boundary value this gives a total `10 km/s` contrast. |
| `--boundary-b-los-gauss FLOAT` | `-500.0` | Finite signed component `B_LOS` at every horizontal boundary. It determines the reference inclination through `arccos(B_LOS/B)`. |
| `--center-b-los-gauss FLOAT` | `500.0` | Finite signed component `B_LOS` at the exact center. Intermediate `B_LOS` values interpolate linearly in the envelope before conversion to inclination. |
| `--field-strength-gauss FLOAT` | `1000.0` | Positive, constant unsigned field magnitude `B`. It must be strictly larger than the absolute value of both LOS endpoints for the generator's nonvertical seed geometry. The field must be positive for a log ratio. |
| `--magnetic-azimuth-deg FLOAT` | `0.0` | Finite, spatially constant azimuth in degrees. The solver treats azimuth as 180-degree periodic and wraps it into its principal interval. It affects the transverse Stokes Q/U orientation but not `B_LOS`. |
| `--spatial-hidden WIDTH[,WIDTH...]` | `96,96,96` | Positive hidden-layer widths for the `(x,y,z)` spatial-correction MLP. Wider/deeper lists increase capacity, memory use, compilation time, and optimizer work. |
| `--spatial-scale T,NE,NH,V,VT,B,GAMMA,CHI` | automatically derived | Optional eight positive finite correction scales, in the stated channel order: dex for T/NE/NH/VT/B and units of 20 km/s, pi, pi/2 for V/GAMMA/CHI. The generator rejects an explicit vector if any truth value is unreachable. Supplying this option disables automatic scale selection. |
| `--spatial-scale-safety-factor FLOAT` | `1.25` | In automatic mode, multiplies the measured maximum required reference-relative correction in each channel. It must be greater than 1. Ignored for scale selection when `--spatial-scale` is supplied, although it is still validated. |
| `--spatial-scale-padding FLOAT` | `0.05` | Non-negative correction margin added after the automatic safety factor. The selected value is the maximum of this padded requirement and the inversion's normal default for each channel. Ignored for scale selection when an explicit scale is supplied. |

The dataset additionally records all contrast endpoints, the horizontal and
thermodynamic envelopes, requested/effective thermodynamic heights, Gaussian
width, temperature anchor, precision, checkpoint seed, required correction
ranges, and selected correction ranges. Consequently, its physical
construction can be audited without relying only on shell history.

A small end-to-end smoke run is:

```bash
python pinn_3d_inversion.py \
  --nx 2 --ny 2 --n-wave 9 \
  --spatial-hidden 16,16 --inversion-epochs 1 \
  --training-batch-columns 2 --wavelength-batch 9 \
  --synthesis-batch-columns 2 --no-progress
```

Spectral inversion uses an optimizer-update-level sin-squared decay. If `s`
is the zero-based optimizer step and `S` is the final step, the rate is
`1e-5 + (1e-3 - 1e-5) sin²[pi/2 (1 - s/S)]`. Thus the first update uses
`1e-3`, the last uses `1e-5`, and the rate is clipped at both endpoints.
Change the endpoints with `--inversion-learning-rate` and
`--inversion-final-learning-rate`. The effective rate at each epoch is stored
in the result archive and logged to W&B.

Weights & Biases tracking is optional. Install the extra, authenticate once,
and add `--wandb` to the normal generation or inversion command:

```bash
python -m pip install -e ".[wandb]"
wandb login

adora-pinn-3d --precision fp32 --invert-only \
  --dataset data/pinn_3d_test_cube.npz \
  --wandb --wandb-project adora-pinn-3d \
  --wandb-name strong-field-fp32 --wandb-tags fp32,strong-field
```

Alternatively, add `--wandb-login` to have the online command call the
interactive `wandb.login()` API immediately before creating the run. No API
key is accepted or stored by Adora itself.

The run records the complete CLI and network configuration, the relative
parameterization, per-epoch training and full-wavelength validation losses,
best validation loss, final full-cube loss, and phase timings.
Per-epoch inversion logging reuses metrics that are already on the host and
does not introduce another GPU synchronization. At epoch zero, every 50
inversion epochs, and the final epoch, W&B also receives a current 4 x 3
evaluation figure: full-Stokes input/current profiles at the center FOV pixel
plus input/current T, ne, Bx, and Bz maps at the middle height. Set a different
cadence with `--wandb-evaluation-every`. Producing these figures evaluates the
current atmosphere cube and synthesizes one complete diagnostic spectrum, so
it intentionally adds work only at those checkpoints.

Use `--wandb-mode offline` on a machine without a network connection. Dataset,
checkpoint, and result archives can be large, so they are only uploaded when
`--wandb-log-artifacts` is supplied. `--wandb-entity`, `--wandb-group`,
`--wandb-dir`, and the other tracking options are listed by `--help`.

The generated archive stores unit-bearing coordinate names, reference and
ground-truth atmosphere cubes, wavelengths, line centers, and observed Stokes
profiles. The result archive stores absolute inferred atmosphere cubes, fitted
spectra, and spectral training/validation histories. It also stores
`inferred_normalized_corrections` with shape `(nx, ny, depth, 8)`, the channel
names, scales, and parameterization metadata so the network outputs can be
inspected directly.

Version-2 checkpoints contain the spatial weights and exact reference profiles.
They are written at initialization and after fitting and support warm starts
with `--checkpoint-in`. The fixed reference is checked against the dataset;
Adam and RNG state restart. Version-1 learned-base checkpoints are incompatible:
start a fresh inversion or regenerate the initial checkpoint. The removed
`--pretrain-*`, `--skip-pretraining`, `--base-*`, and `--fine-tune-base` flags
are rejected instead of being silently ignored. Existing physical dataset
archives remain usable, subject to the new correction ranges.

Atmospheric fields use shape `(nx, ny, depth)` and Stokes data use
`(nx, ny, I/Q/U/V, wavelength)`.

To also write a compact spectra-only archive for downstream tools, provide an
output path (or omit the path to use `data/pinn_3d_spectra.npz`):

```bash
adora-pinn-3d --invert-only \
  --dataset data/pinn_3d_test_cube.npz \
  --spectra-output data/my_run_spectra.npz

# Uses the default spectra filename:
adora-pinn-3d --invert-only --spectra-output
```

That NPZ contains `wavelength_nm`, `x_normalized`, `y_normalized`,
`stokes_labels`, `input_stokes`, and `output_stokes`. The two Stokes arrays
have shape `(nx, ny, 4, n_wave)`.

Generated truth cubes are checked for physical validity and against the
reference-relative correction ranges before synthesis. This prevents
a custom perturbation from silently producing an inversion target the selected
neural parameterization cannot represent.

This is a 3-D neural representation with independent vertical rays: the LTE
transfer is 1.5-D and does not include horizontal radiative coupling. The two
Fe I lines cannot uniquely determine eight quantities at every height, so the
fixed reference, shared smooth representation, bounded corrections,
and optional `--prior-weight` are important regularization. A full `50 x 50`
run is computationally substantial on CPU.

For accelerator tuning, memory use is driven primarily by the number of
columns and wavelengths evaluated concurrently. `--training-batch-columns`
sets the training column batch, `--wavelength-batch` sets how many wavelengths
are sampled for each optimizer update, and `--wavelength-parallelism` bounds
how many of those wavelengths run concurrently. The default parallelism is
48; lower it first if training runs out of device memory, or raise it toward
the active wavelength count when memory is available and profiling shows more
throughput is possible. Values above `--wavelength-batch` do not add training
parallelism. For truth-cube generation and final rendering,
`--synthesis-batch-columns` independently controls the number of concurrent
columns. In practice, tune the column count and wavelength parallelism
together because their product dominates the live radiative-transfer work.

JAX can reuse compiled executables across processes when a persistent cache is
configured. Point the cache at a durable, writable directory before running
the same shapes repeatedly:

```bash
export JAX_COMPILATION_CACHE_DIR=/absolute/path/to/adora-jax-cache
python pinn_3d_inversion.py --invert-only --dataset data/pinn_3d_test_cube.npz
```

Changing batch shapes or `--wavelength-parallelism` creates a different
compiled variant, so benchmark steady-state epochs after the first compile.

### Comparing FP32 and FP64 on an RTX 4090

The comparison command launches fresh processes because JAX precision is a
process-global setting. Both workers load the same FP64-created initial
checkpoint and use the same sampled column/wavelength schedule. It compares
continuum-normalized spectra before and after inversion, the initial loss and
gradient (including relative L2 error and cosine similarity), complete
inversion/validation histories, and all eight recovered atmosphere
fields. First-call and synchronized cached timings are reported separately.

Generate a reference observation cube once, then run the paired benchmark on
the 4090:

```bash
adora-pinn-3d --precision fp64 --generate-only \
  --dataset data/pinn_3d_test_cube.npz

adora-compare-pinn-precision \
  --dataset data/pinn_3d_test_cube.npz \
  --require-device-substring 4090 \
  --output data/pinn_precision_comparison.json
```

The JSON report contains `speedup_fp64_over_fp32` for the spectrum, gradient,
production inversion and total run, plus paths to both numerical
artifact archives and fitted results. Values above one mean FP32 was faster.
GPU execution is required by default; `--platform cpu --allow-cpu` exists only
for plumbing and numerical smoke tests. Use `--overwrite` to replace a prior
report and its generated artifacts.

The small one-epoch command above checks compilation and data flow. Scientific
recovery still requires adequate epochs and sufficient spectral information;
normalizing outputs does not remove degeneracies between atmospheric fields.

### Visualizing the 3-D inversion

`visualize_pinn_3d_inversion.py` reads the self-contained result archive and
writes four headless figures: input and inferred thermodynamics, followed by
input and inferred Cartesian magnetic components. Every figure has three
height rows and covers the complete normalized horizontal domain. The
thermodynamic figures have four columns; the magnetic figures have three.
Every subplot has an individual colorbar spanning the 5th to 95th percentile
of its corresponding input/truth horizontal slice. The matching inferred
subplot uses exactly the same truth-derived limits, so inversion outliers
cannot rescale the comparison; values outside the central 90% saturate at the
ends of the colormap.

```bash
python visualize_pinn_3d_inversion.py \
  --result data/pinn_3d_inversion.npz \
  --output-dir figures/pinn_3d
# or, after installation:
adora-plot-pinn-3d
```

The default rows are nearest to 20%, 50%, and 80% of the geometric height
range. Select physical heights or exact depth indices instead with, for
example:

```bash
python visualize_pinn_3d_inversion.py --heights-km 250 500 1000
python visualize_pinn_3d_inversion.py --height-indices 18 24 35 --format pdf
```

Temperature is shown in kelvin, electron and total hydrogen densities as
`log10(m^-3)`, and LOS velocity in `km/s`. The magnetic panels convert the
archived strength,
inclination, and azimuth into `Bx = B sin(gamma) cos(chi)`,
`By = B sin(gamma) sin(chi)`, and LOS `Bz = B cos(gamma)`, displayed in gauss.
Use `--state` or `--quantity` to render only a subset.

### Interactive spectra and atmosphere comparison

`compare_inversion_gui.py` provides the linked 4 x 3 GUI. Column one contains
Stokes I, Q, U, and V (one row each), with input and inverted profiles
overplotted. Columns two and three contain input and inverted horizontal maps;
their rows are temperature, `log10(ne)`, Bx, and Bz. Click any map to select an
`(x, y)` location and update all four spectra. Move the bottom slider to select
the atmospheric height shown by all eight maps. Each input/output map pair
shares input-derived color limits.

Open a normal self-contained inversion result with:

```bash
adora-view-pinn-3d --result data/pinn_3d_inversion.npz
# or
python compare_inversion_gui.py --result data/pinn_3d_inversion.npz
```

The GUI also has `Load result…` and `Load 4 files…` buttons. To specify the
four files at launch instead, use:

```bash
adora-view-pinn-3d \
  --input-spectra data/input_spectra.npz \
  --output-spectra data/output_spectra.npz \
  --input-quantities data/input_atmosphere.npz \
  --output-quantities data/inverted_atmosphere.npz
```

Spectra files need the coordinate/wavelength axes and either `stokes` or the
state-specific `input_stokes`/`output_stokes` key. Quantity files need the
coordinate/height axes, temperature and electron density, plus either Bx/Bz
in gauss or magnetic strength (tesla), inclination, and azimuth. The native
`truth_` and `inferred_` field names written by the inversion are recognized
directly. `--spectra` and `--quantities` are shortcuts when each state pair is
stored together.

### Known scientific limitations

- The continuum model is deliberately incomplete and there is no EOS solve.
- Van der Waals broadening uses the simple Kurucz coefficient; an Unsöld/ABO
  treatment remains future work.
- Fe ionization presently includes stages I--III only.
- This remains a compact research code, not a general stellar-atmosphere
  inversion package. Numerical regression tests do not replace validation for
  a new atmosphere, line list, geometry, or observing setup.
