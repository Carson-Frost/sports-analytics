# Sports Analytics

These are my NBA and NFL analyses: end-of-game decisions, fourth-down decisions and offensive
personnel. Each project below is its own folder with its code, a write-up of what I found, and
its charts.

## Projects

### [NBA Fouling Up 3](nba-foul-up-3/)

A possession-by-possession model of the last 20 seconds of NBA games, 2021-22 to 2025-26. A team up
3 should foul with 2 to 5 or 6 seconds left, though it's risky that late, and defend from 14 or 15
seconds on, depending on whether the trailing team has a timeout. In between, neither choice is
clearly better. Wins count only at the buzzer. Teams up 3 now foul on about a quarter of these
possessions, up from 7% in 1996-97.

Python, pandas, numpy, matplotlib

### [NFL Fourth-Down Decision Model](nfl-fourth-down-model/)

Scores going for it, kicking and punting on every regular-season fourth down from 2015 to 2025,
using nflfastR's win-probability model. When the model clearly favored going for it, teams went for
it 27% of the time in 2015 and 54% in 2025. Includes a calculator for any fourth-down situation.

Python, pandas, numpy, scikit-learn, xgboost, matplotlib

### [NFL Personnel Groupings](nfl-personnel-efficiency/)

How often every offense uses each personnel grouping, and the EPA per play of each, from per-play
personnel on 327,857 regular-season runs and dropbacks, 2016 to 2025. For the same offense in the
same down, distance and field position, no grouping was clearly more efficient than 11 personnel
overall. Dropbacks out of 12 personnel gain 0.038 more EPA than dropbacks out of 11 personnel, and
runs out of 12 personnel gain 0.042 less.

Python, pandas, numpy, matplotlib

## Data

Everything here uses **public data**:

- **NBA.com Stats**, used under NBA.com's terms of use, read through the sportsdataverse
  hoopR-nba-stats-raw archive (CC BY 4.0).
- **nflverse** play-by-play, CC BY 4.0, and personnel from NFL Next Gen Stats and FTN Data via
  nflverse, CC BY-SA 4.0.
- **nflfastR's win-probability model**, from the fastrmodels R package (MIT).

No project commits raw data; each one's `scripts/pull.py` downloads it. Each project's README
names its exact source and what it commits.
