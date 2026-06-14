# Backtesting Report

How this system *would have* predicted the 2014, 2018 and 2022 World Cups, trained only on matches played before each tournament and using only ratings available before each match (a leakage-safe walk-forward Elo). Predictions are scored on the 64 matches of each tournament.

## Match-prediction accuracy

| tournament | matches | log loss | Brier | accuracy | calibration error |
| --- | --- | --- | --- | --- | --- |
| 2014 World Cup | 64 | 0.9232 | 0.5428 | 0.625 | 0.1191 |
| 2018 World Cup | 64 | 0.9516 | 0.5632 | 0.5625 | 0.0214 |
| 2022 World Cup | 64 | 1.0276 | 0.6054 | 0.5312 | 0.1248 |

Across the three tournaments: mean log loss **0.9675**, mean Brier **0.5705**, mean accuracy **0.5729**, mean calibration error **0.0885**. For reference an uninformed 1/3-each model scores log loss ~1.099.

## Did we fancy the eventual champion?

| tournament | champion | predicted rank | top 3 | top 5 | top 10 |
| --- | --- | --- | --- | --- | --- |
| 2014 World Cup | Germany | 4/32 | no | yes | yes |
| 2018 World Cup | France | 5/32 | no | yes | yes |
| 2022 World Cup | Argentina | 2/32 | yes | yes | yes |

*Rank is the champion's position in the pre-tournament Elo favourite ordering among the 32 participants — a proxy for a full winner-probability simulation.*

## Caveats

- The champion **rank** uses the pre-tournament Elo favourite ordering, not a Monte-Carlo winner-probability simulation: that would require each year's official group draw and bracket encoded as a tournament config, which are **not implemented** for historical years (TODO). The simulation hook exists but is unused here — we do not fabricate a draw.
- Football is high-variance and knockouts are short series; a strong model can still rank the eventual winner outside the top few. These are small samples (3 tournaments, 64 matches each).
- No hyperparameters were tuned on the tournament being scored.
