# Sports Analytics

These are my NBA and NFL analyses: end-of-game choices, fourth-down choices and offensive
personnel. Each project below is its own folder with its code, a write-up of what I found, and
its charts.

## Projects

### [NBA Fouling Up 3](nba-foul-up-3/)

A possession-by-possession model of whether a team up 3 in the last 20 seconds of an NBA game
should foul or defend, 2021-22 to 2025-26.

Python, pandas, numpy, matplotlib

### [NFL Fourth-Down Calls](nfl-fourth-down-model/)

Scores going for it, kicking and punting on every regular-season fourth down from 2015 to 2025,
using nflfastR's win-probability model. Includes a calculator for any fourth-down situation.

Python, pandas, numpy, scikit-learn, xgboost, matplotlib

### [NFL Personnel Groupings](nfl-personnel-efficiency/)

How often every offense uses each personnel grouping, and the EPA per play of each, from per-play
personnel on regular-season runs and dropbacks, 2016 to 2025.

Python, pandas, numpy, matplotlib

## Layout

Each analysis is one self-contained folder: its own Python package, tests, data pipeline and
write-up, with nothing shared between folders. Every folder has the same shape:

| Path | Holds |
| --- | --- |
| `README.md` | The write-up: question, method, findings, charts, and the exact run steps |
| `pyproject.toml` | The package, its dependencies and its `dev` extras (pytest, ruff) |
| `requirements.lock` | Every dependency pinned to one version, so results and figures come out the same on any machine |
| `src/` | The analysis code the scripts call |
| `scripts/` | The pipeline, run in order: `pull.py` downloads the raw data, then the build, analysis and `figures.py` steps, and `readme.py` writes the results into the README |
| `data/` | Downloaded and derived data; raw data is not committed |
| `results/` | The tables the analysis writes |
| `figures/` | The charts |
| `fonts/` | Liberation Sans (SIL OFL), the font every chart is drawn in |
| `tests/` | The pytest suite |

`nfl-fourth-down-model/` also has `models/`, the win-probability model that
`scripts/get_wp_model.py` fetches.

To run an analysis, work inside its folder:

```bash
cd <analysis>
pip install -r requirements.lock -e ".[dev]"
pytest -q && ruff check .
```

Then run its `scripts/` in the order its README's "Run it" section gives. Some steps take a long
time. CI runs `ruff check .` and `pytest -q` in each folder on Python 3.12.

## Data

Everything here uses **public data**:

- **NBA.com Stats**, used under NBA.com's terms of use, read through the sportsdataverse
  hoopR-nba-stats-raw archive (CC BY 4.0).
- **nflverse** play-by-play, CC BY 4.0, and personnel from NFL Next Gen Stats and FTN Data via
  nflverse, CC BY-SA 4.0.
- **nflfastR's win-probability model**, from the fastrmodels R package (MIT).

No project commits raw data; each one's `scripts/pull.py` downloads it. Each project's README
names its exact source and what it commits.
