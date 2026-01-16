# Reproducing Paper Figures

This repo contains scripts used to generate the simulations and plots from
`docs/Byzantine_Computing_on_Unknown_Data.pdf`. The exact figure mapping
below is a best-effort outline based on the current code.

## Multiplicative Weights simulations
- **Algorithm 1 (pseudocode)**: Implemented in `MW.py` (`run_sim`).
- **Blacklisting test (Figure 1)**: Run `MW.py` with `kill_byz=1` and a
  small `threshold`. The `blacklist_test` helper at the bottom is the
  intended entry point.
- **Estimator loss curves (Figures 2–3)**: `MW.py` produces
  `error_evolution_mom_guided.png` and `trajectories_mom_guided.png`.
  Use different loss settings and `kill_byz` to match the paper variants.

## MW convergence vs theory
- **Empirical vs theoretical T (Figures 4–7, Table I)**:
  `verification.py` sweeps `eta` and `tau`, writes a CSV, and produces
  `empirical_vs_theoretical_T.png`, `convergence_best_fit.png`,
  `error_histogram.png`, and `heatmaps.png` under `mw_weight_plots/`.

## Notes
- Set the random seed at the top of each script for deterministic runs.
- Output directories are defined inside each script (search for `output_dir`).
- Manim animations are not used for the paper plots, but can be rendered via
  `manim` for presentation visuals.
