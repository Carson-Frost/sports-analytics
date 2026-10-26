"""Write README.md from the results and an example run of scripts/call.py."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
EXAMPLE = ["--distance", "2", "--spot", "opp 45", "--quarter", "4", "--clock", "2:00", "--score", "-3"]


def main():
    s = pd.read_csv(R / "by_season.csv").set_index("season")
    first, last = s.index.min(), s.index.max()
    a, b = s.loc[first], s.loc[last]
    g = pd.read_csv(R / "graded_fourth_downs.csv.gz")
    went = g.coach == "go"
    rate = {y: 100 * went[g.season == y].mean() for y in (first, last)}
    example = subprocess.run([sys.executable, ROOT / "scripts" / "call.py", *EXAMPLE],
                             capture_output=True, text=True, check=True).stdout.strip()

    text = f"""# NFL Fourth-Down Decisions

NFL teams go for it on fourth down far more than they used to: on {rate[first]:.0f}% of fourth downs in
{first} and {rate[last]:.0f}% in {last}, a shift credited to teams adopting analytics
([Sportico](https://www.sportico.com/leagues/football/2024/detroit-lions-dan-campbell-nfl-stats-fourth-down-1234817656/)).
By this project's model, they still don't go for it enough. When the model clearly favored going for
it, teams went for it {a.go_rate:.0f}% of the time in {first} and {b.go_rate:.0f}% in {last}.

![Share of fourth downs where the model clearly favored going for it and the team went](figures/go_rate.png)

## Method

```python
wp = WinProbability()  # nflfastR's wp_model_spread, averaged over ±1 minute of clock

go   = p_convert * wp(our_first_down_after_gain) + (1 - p_convert) * wp(their_ball_here)
fg   = p_make * wp(up_3_their_kickoff_return) + (1 - p_make) * wp(their_ball_at_kick_spot)
punt = mean(wp(their_ball_after_each_punt))
```

Conversion, gains, field goals, punts, kickoffs and clock are fit on {first}-{last} plays. Each fourth
down is graded again on 200 resamples of the games. I set the bar for a clear call at
1 percentage point for simplicity: the better option has to win by that much in the middle 90% of
resamples.

## Results

![The model's best call by distance and field position](figures/recommendation_map.png)

## Fourth-Down Calculator

Example: 4th and 2 at the opponent's 45, down 3, with 2:00 left in the 4th quarter:

```bash
python scripts/call.py {" ".join(f'"{x}"' if " " in x else x for x in EXAMPLE)}
```

```
{example}
```

Optional: `--timeouts`, `--opp-timeouts`, `--spread`, `--home`, `--gets-second-half-kickoff`.

## Limits

- Every team is treated as average except for the point spread.
- nflfastR's model may have been trained on the seasons graded here.
- The intervals don't include error in the win-probability model.

## Run it

```bash
pip install -e ".[dev]"
python scripts/pull.py
python scripts/get_wp_model.py
python scripts/build.py
python scripts/grade.py      # about an hour
python scripts/figures.py
python scripts/readme.py
pytest && ruff check .
```

## Credits

[nflverse](https://github.com/nflverse/nflverse-data) play-by-play (CC BY 4.0). nflfastR's
win-probability model from [fastrmodels](https://cran.r-project.org/package=fastrmodels) (MIT).
Method from [nfl4th](https://github.com/nflverse/nfl4th) and
[Romer (2006)](https://doi.org/10.1086/501171).
"""
    (ROOT / "README.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
