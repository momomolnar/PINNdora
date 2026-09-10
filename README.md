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
  FAL-C pretraining, and spectropolarimetric inversion.
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
2. Pretrains a Fourier-height neural field on FAL-C. A second, initially zero,
   network represents bounded spatial corrections as a function of `(x,y,z)`,
   so every horizontal column initially returns the same FAL-C atmosphere.
3. Fits that shared spatial network with continuum-normalized, weighted
   Stokes residuals. Complete columns are shuffled into fixed-size batches;
   wavelength subsets are sampled uniformly, making each batch loss an
   unbiased estimate of the full-spectrum objective. Every Kurucz line center
   is included exactly in the saved wavelength grid. A fixed full-wavelength
   validation batch selects the best epoch, and fitting fails explicitly if no
   epoch improves on the pretrained field.

On a GPU or other accelerator, the independent columns inside each synthesis
batch are vectorized. Wavelengths are processed in sequential chunks, with the
wavelengths inside each chunk vectorized together; the default
`--wavelength-parallelism 48` matches the default training wavelength batch.
CPU execution retains the faster sequential column path. The PINN synthesis
also uses the nonsingular DELO specialization permitted by its bounded,
positive atmosphere decoder. This keeps the generic solver's expensive matrix
exponential fallback out of the batched differentiation graph; the generic
solver remains available for atmospheres that may contain singular opacity.

FAL-C pretraining is compiled as one `jax.lax.scan`, and all ordered optimizer
updates in each spectral epoch run in another compiled scan. Loss histories
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

A small end-to-end smoke run is:

```bash
python pinn_3d_inversion.py \
  --nx 2 --ny 2 --n-wave 9 \
  --base-hidden 16,16 --spatial-hidden 16,16 --base-frequencies 3 \
  --pretrain-steps 50 --pretrain-tolerance 1.0 --inversion-epochs 1 \
  --training-batch-columns 2 --wavelength-batch 9 \
  --synthesis-batch-columns 2 --no-progress
```

The generated archive stores unit-bearing coordinate names, reference and
ground-truth atmosphere cubes, wavelengths, line centers, and observed Stokes
profiles. The result archive stores inferred atmosphere cubes, fitted spectra,
and both loss histories; a separate named-array checkpoint stores both neural
networks. The checkpoint is written after pretraining and again after fitting;
it can be used as a warm start with `--checkpoint-in ... --skip-pretraining`.
This restarts Adam and is not an exact optimizer/RNG continuation. Atmospheric
fields use shape `(nx, ny, depth)` and Stokes data use
`(nx, ny, I/Q/U/V, wavelength)`.

Generated truth cubes are checked against both the physical decoder bounds and
the frozen spatial network's correction range before synthesis. This prevents
a custom perturbation from silently producing an inversion target the selected
neural parameterization cannot represent.

This is a 3-D neural representation with independent vertical rays: the LTE
transfer is 1.5-D and does not include horizontal radiative coupling. The two
Fe I lines cannot uniquely determine eight quantities at every height, so the
pretrained field, shared smooth representation, bounded physical transforms,
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
pretraining/inversion/validation histories, and all eight recovered atmosphere
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
pretraining, production inversion, and total run, plus paths to both numerical
artifact archives and fitted results. Values above one mean FP32 was faster.
GPU execution is required by default; `--platform cpu --allow-cpu` exists only
for plumbing and numerical smoke tests. Use `--overwrite` to replace a prior
report and its generated artifacts.

The shortened 50-step command above is a compilation and plumbing smoke test,
not a scientifically adequate pretraining run. The default workflow restores
the best pretraining iterate and requires its normalized FAL-C latent MSE to
reach `1e-4` before spectral fitting proceeds.

### Visualizing the 3-D inversion

`visualize_pinn_3d_inversion.py` reads the self-contained result archive and
writes four headless figures: input and inferred thermodynamics, followed by
input and inferred Cartesian magnetic components. Every figure has three
height rows and covers the complete normalized horizontal domain. The
thermodynamic figures have four columns; the magnetic figures have three.
Input/output figures use the same component color limits.

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

### Known scientific limitations

- The continuum model is deliberately incomplete and there is no EOS solve.
- Van der Waals broadening uses the simple Kurucz coefficient; an Unsöld/ABO
  treatment remains future work.
- Fe ionization presently includes stages I--III only.
- This remains a compact research code, not a general stellar-atmosphere
  inversion package. Numerical regression tests do not replace validation for
  a new atmosphere, line list, geometry, or observing setup.
