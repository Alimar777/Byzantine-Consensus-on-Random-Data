# Byzantine Consensus using Multiplicative Weights on Sub-Gaussian External Data

This repo is a research/visualization sandbox around Byzantine-robust
aggregation, multiplicative weights (MW) updates, and consensus sketches.
It mixes simulation code (NumPy/Matplotlib) with Manim animations that
illustrate the same ideas.

Goals:
- Explore robust estimators under adversarial samples.
- Visualize MW dynamics and consensus intuitions.
- Generate figures/animations for talks or papers.

## What's inside
- `MW.py`: Core simulation for robust estimation under Byzantine samples.
  Implements straight mean, MW mean, MoM-guided MW, and median-of-means,
  with multiple loss variants and optional blacklisting.
- `verification.py`: Parameter sweeps and verification plots for MW
  blacklisting bounds; produces CSVs and heatmaps.
- `consensus_oracle_sim.py`: Manim animation of oracle querying and
  threshold consensus.
- `consensus_oracle_sim_2.py`: Extended consensus animation with
  committee sampling and Byzantine behavior.
- `consensus_bug_check.py`: Grid search over MW hyperparameters and
  theoretical vs empirical convergence plots.
- `mw_animation.py`: Manim animation of MW weight updates over rounds.
- `wisdom_of_crowd.py`: Manim animation comparing mean/median/MoM under
  adversarial guesses.
- `dual.py`: Small duality visualization (primal/dual segments and vectors).

## Setup
Python 3.10+ is recommended. Main dependencies used across scripts:
- numpy
- matplotlib
- pandas
- manim (for animations)
- scipy (for curve fitting in `consensus_bug_check.py`)
- scikit-learn (for R^2 in `consensus_bug_check.py`)

If you only need simulations (no animations), you can skip Manim.

System deps for Manim (animations only):
- ffmpeg
- LaTeX (e.g., MiKTeX or TeX Live)

Install:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start
Run a single simulation and plots:
```bash
python MW.py
```

Run verification sweeps (this can be slow; uses multiprocessing):
```bash
python verification.py
```

Run an animation (example):
```bash
manim -pqh mw_animation.py MultiplicativeWeightsAnimation
```

## Usage guide by script
- `MW.py`: tweak data generation, loss choice, and blacklisting in the
  parameter block near the bottom. Produces error/trajectory plots.
- `verification.py`: runs multi-trial sweeps and saves CSVs/plots under
  `output_dir/verification`.
- `consensus_oracle_sim*.py`: Manim scenes for oracle querying/committee
  behavior; edit parameters at top for node counts and thresholds.
- `consensus_bug_check.py`: grid searches eta/tau and compares empirical vs
  theoretical convergence of MW updates.
- `mw_animation.py` and `wisdom_of_crowd.py`: presentation-ready Manim
  animations; render via `manim`.
- `dual.py`: small geometry duality illustration that saves PNGs.

## Outputs
Most scripts save plots or videos to local output folders:
- MW plots: see the `output_dir` configured in `MW.py`.
- Verification plots/CSVs: subfolders under `output_dir/verification`.
- Manim output: `media/` (videos, images, LaTeX renders).

Note: `media/` is gitignored except for PDFs. Regenerate videos as needed.

## Notes
- The code is research-grade and optimized for exploration/visuals rather
  than packaging. Expect tunable parameters at the top of scripts.
- Some runs can be slow (multiprocessing sweeps, Manim renders).

## Paper
See `docs/Byzantine_Computing_on_Unknown_Data.pdf`.

## Reproducing figures
See `docs/reproduce_figures.md`.

## Citations / references
References and citations are listed in the paper. If you publish results
based on those ideas, cite the original sources.

## Authors
The associated paper lists: Kelsey Knowlson, Victoria Lien, Bryce Palmer,
and Matthew Rackley.

## License
All rights reserved. See `LICENSE.md`.
