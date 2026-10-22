# Sports Analytics

These are my NBA and NFL analyses: player ratings, draft pick value, end-of-game and fourth-down
decisions, overtime strategy and offensive personnel. Each project below is its own folder
with its code, a write-up of what I found, and its charts.

## Projects

### [Exploring NBA Regularized Adjusted Plus Minus](nba-adjusted-plus-minus/)

Regularized adjusted plus-minus for every player in the 2025-26 regular season, from the
play-by-play of all 1,230 games. Among players with 1,000 minutes, how good a player's team was
explains 77% of the differences in raw plus-minus and only 19% of the adjusted version, so raw
plus-minus is mostly the team. Victor Wembanyama comes out on top at +7.6 points per 100
possessions better than an average player, but not by much: he is first in only 48 of 200
resampled seasons.

Python, pandas, numpy, scipy, matplotlib

### [NBA Draft Pick Value Curve](nba-draft-pick-value/)

A curve of what each draft pick is worth, fit to the career Win Shares of all 1,568 players
drafted from 1990 to 2016. The No. 1 pick is worth 74.9 career Win Shares, the value is cut in
half by pick 8, and every time the pick number doubles, the pick loses about 12.5 Win Shares. On
paper the No. 1 pick is worth two No. 8 picks.

Python, pandas, numpy, lxml, matplotlib

### [NBA Fouling Up 3](nba-foul-up-3/)

A play-by-play model of the last 24 seconds of NBA games, 2020-21 to 2025-26. When a team up 3
hands the other team the ball, fouling is the better choice with 5 seconds or less left and
defending with 15 or more. Yet teams foul most where it barely matters: 36.4% of the time with 9
to 12 seconds left, against 1.3% with 3 or less, where it pays most.

Python, nba_api, pandas, numpy, matplotlib

### [NFL Fourth-Down Decision Model](nfl-fourth-down-model/)

Scores going for it, kicking and punting on every regular-season fourth down from 2015 to 2025,
using nflfastR's win-probability model. When the model clearly favored going for it, teams went for
it 27% of the time in 2015 and 54% in 2025. Includes a calculator for any fourth-down situation.

Python, pandas, numpy, scikit-learn, xgboost, matplotlib

### [NFL Overtime: Kick or Receive?](kick-or-receive/)

A snap-by-snap simulation, built from 1,201,470 plays, that plays 100,000 overtimes under each
rule the league has used. Under today's rule, where both teams get the ball, the coin-toss winner
should still take it: receiving wins 53.6% of playoff overtimes. Under the old sudden death rule,
played on today's game, it won 64.5%, so receiving is still right but the edge has shrunk.

Python, pandas, numpy, matplotlib

### [NFL Personnel Groupings](nfl-personnel-efficiency/)

How often every offense uses each personnel grouping, and the EPA per play of each, from per-play
personnel on 327,857 regular-season runs and dropbacks, 2016 to 2025. For the same offense in the
same down, distance and field position, no grouping was clearly more efficient than 11 personnel overall.
Dropbacks out of 12 personnel gain 0.038 more EPA than dropbacks out of 11 personnel, and runs out of 12 personnel
gain 0.042 less.

Python, pandas, numpy, matplotlib

## Data

Everything here uses **public data**:

- **NBA.com Stats**, used under NBA.com's terms of use, read through the sportsdataverse
  hoopR-nba-stats-raw archive (CC BY 4.0) or `nba_api` (MIT).
- **Basketball Reference** draft pages, used under Sports Reference's terms of use.
- **nflverse** play-by-play, CC BY 4.0, and personnel from NFL Next Gen Stats and FTN Data via
  nflverse, CC BY-SA 4.0.
- **nflfastR's win-probability model**, from the fastrmodels R package (MIT).
- **Lee Sharpe's nfldata** schedule, which has no licence file, so it isn't redistributed.

No project commits raw data; each one's `scripts/pull.py` downloads it. Each project's README
names its exact source and what it commits.
