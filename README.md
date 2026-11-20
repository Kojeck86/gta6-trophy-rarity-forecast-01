# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kojeck86/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kojeck86/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on [Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95) per trophy,
scored by pinball loss on the logit scale.

## How it works

Exophase only moves its counts when it refreshes a game's page, so a case
history is a step function: long flat stretches, then a jump in
players-tracked and in every trophy's holder count. The forecast is built in
holder space, not rate space (`main.py`, Python 3.13 standard library only):

- **Players-tracked** is extrapolated to the target by a least-squares fit of
  log N against log(1 + hours since release) over the recent distinct reads,
  capped at 20x the last count.
- **Holder counts** grow by two terms, fitted per trophy from the history's
  refreshes (recency-weighted least squares, shrunk toward the tier mean):
  - `alpha` per newly tracked player — the trophies added at a refresh
    per new player;
  - `beta` per hour per tracked player — the existing players' progress,
    which keeps adding holders even when tracking growth saturates. This
    term is what separates the forecast from a persistence baseline over a
    4.5-day horizon: by the end of launch week the tracked base flattens
    while players keep finishing the game.
- The median rate at the target is `E(target) / N(target)`.
- The **spread** is a mixture, not a plain Gaussian: with probability `p`
  (from the refresh rate in the history, or a 4-hour-gap prior when the
  history is still flat) a refresh lands before the target and the rate
  follows the cohort forecast with a logit spread calibrated by
  leave-one-out prediction of the history's own refreshes; with
  probability `1-p` no refresh lands and the rate stays exactly where the
  last read put it, because the page is frozen between refreshes. The
  spread is pooled across all trophies at once, never tuned per trophy.

When the history is still flat (the practice cases), the parameters fall
back to priors calibrated on the launch-night refresh the competition
describes: the first refresh took players-tracked from 638 to 3,868 while
trophies-earned went 943 to 4,276, so a fresh player holds the
non-universal trophies at about 6.7% of the old cohort's per-player rate,
and the universal trophy (100%) stays universal.

## Running it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The entrypoint is `main.py` (Python 3.13 standard library only); there is no
`main.ts` in this repository. Reads every `*.in` case in the input directory
and writes one `<id>.out` per case. Deterministic (two runs give
byte-identical output), about 0.1 s per case, no network, no randomness.

## Local validation

The practice answers are not in the kit, so the practice is validated
against answers reconstructed from the aggregates the competition
publishes (refresh 638 -> 3,868 players, 943 -> 4,276 trophies earned,
every rate but the 100% one fell) under three distributions of the new
cohort's trophies; the house persistence baseline is re-implemented from
the rules text for comparison. The final is validated on two synthetic
player-level simulations of a launch week (constant hazard, and saturating
playtime), which only exercise the multi-refresh path — they do not prove
accuracy on the real final.

| check | this entry | house baseline |
| --- | --- | --- |
| practice, proportional split | 0.802 | 1.401 |
| practice, uniform split | 0.793 | 1.387 |
| practice, early-trophy split | 0.872 | 1.517 |
| synthetic final, constant hazard | 0.211 | 0.785 |
| synthetic final, saturating playtime | 0.120 | 0.369 |

Loss is the mean pinball loss on the logit scale (lower is better). The
house's own stated practice loss is about 1.51, which the reconstructed
answers reproduce (1.39–1.52). The flat-history leave-one-out check scores
0.027 (baseline 0.014): both near zero, as expected when nothing moves —
the small premium is the refresh insurance the mixture buys.

## License

MIT — see `LICENSE`.
