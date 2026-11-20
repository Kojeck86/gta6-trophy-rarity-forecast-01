#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase.

Reads one case file per run (makefun/gta6-trophies-case/v1) from the input
directory and writes, for each trophy, five quantiles (5/25/50/75/95) of its
earn rate at the case target, in percent.

Method, in brief. Exophase only moves its counts when it refreshes a game's
page, so a case history is a step function: long flat stretches, then a jump
in players-tracked and in every trophy's holder count. The forecast is built
in holder space, not rate space:

  * players-tracked N(t) is extrapolated to the target by a least-squares fit
    of log N against log(1 + hours since release) over the recent distinct
    reads, capped at 20x the last count;
  * each trophy's holder count E(t) is grown by two terms, fitted per trophy
    from the history's refreshes and shrunk toward the tier mean:
      - alpha x (new players tracked): the trophies added per newly tracked
        player at a refresh. Alpha bundles the new cohort's holdings and
        cannot be separated from progress on a single refresh, so it is
        fitted jointly with:
      - beta x (hours since the last read) x (players tracked then): the
        existing players' progress, which keeps adding holders even when
        tracking growth saturates. Without this term a forecast would miss
        the launch week's real motion, which is existing players finishing
        the game while the tracked base flattens;
  * the median rate at the target is E(target) / N(target).

The predictive distribution is a mixture, because a refresh may or may not
land before the target: with probability p (from the refresh rate seen in
the history, or a 4-hour-gap prior when the history is still flat) at least
one refresh lands and the rate follows the cohort forecast with a logit
spread calibrated by leave-one-out prediction of the history's own refreshes;
with probability 1-p no refresh lands and the rate stays exactly where the
last read put it (the page is frozen between refreshes). Quantiles come from
inverting that mixture's CDF by bisection. The spread is pooled across all
trophies at once, never tuned per trophy.

When the history is still flat (the practice cases), alpha and beta fall back
to priors calibrated on the launch-night refresh the competition itself
describes: the first refresh took players-tracked from 638 to 3,868 while
trophies-earned went 943 to 4,276, so a fresh player holds the
non-universal trophies at about 6.7% of the old cohort's per-player rate,
and the universal trophy (100%) stays universal.

Standard library only. Deterministic: no randomness, no wall-clock reads
apart from the case file itself, fixed iteration counts.
"""

import json
import math
import os
import sys

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]
RELEASE = "2026-11-19T00:00:00Z"

# Priors for a history that has not seen a refresh yet.
PRIOR_REFRESH_GAP_HOURS = 4.0      # Exophase refreshes hours apart
PRIOR_JUMP = 4.0                   # players-tracked multiplier at a refresh
PRIOR_FRESH_FACTOR = 0.067         # fresh cohort's per-trophy rate, as a
                                   # fraction of the old cohort's rate
PRIOR_SPREAD = 0.9                 # logit spread when no refresh was seen
SPREAD_MIN = 0.45
SPREAD_MAX = 1.50
N_GROWTH_CAP = 20.0
SHRINK_KAPPA = 300.0               # pseudo-count (players) for alpha shrinkage
RECENCY_DECAY = 0.8                # per-refresh weight decay for alpha


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(rate_pct):
    """Logit of a percent rate, clamped so 0 and 100 have a logit."""
    p = clamp(rate_pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(logit):
    """Inverse logit, back to a percent in [0.01, 100]."""
    if logit >= 0:
        p = 1.0 / (1.0 + math.exp(-logit))
    else:
        e = math.exp(logit)
        p = e / (1.0 + e)
    return clamp(p * 100.0, 0.01, 100.0)


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into epoch seconds."""
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def hours_since_release(epoch_seconds):
    return max(0.0, (epoch_seconds - parse_time(RELEASE)) / 3600.0)


def distinct_states(history):
    """Collapse consecutive identical reads into refresh states.

    Returns a list of dicts {t (epoch s), n (players tracked),
    counts (holder counts per trophy, recovered from the 2-decimal rates)}.
    """
    states = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        n = read.get("players_tracked")
        if n is None or not rates:
            continue
        key = (n, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        counts = []
        for r in rates:
            if r is None:
                counts.append(None)
            else:
                counts.append(int(round(r / 100.0 * n)))
        states.append({
            "t": parse_time(read["at"]),
            "n": n,
            "counts": counts,
        })
    return states


def fit_log_log(xs, ys):
    """Least-squares line y = a + b*x. Returns (a, b)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return ys[0], 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def forecast_players(states, target_epoch):
    """Extrapolate players-tracked to the target epoch.

    Log-log fit over the recent distinct reads (last 24 h when there are
    enough), extrapolated and capped. Returns (n_target, refresh_rate_per_h).
    """
    last = states[-1]
    n_last = last["n"]
    span_h = (last["t"] - states[0]["t"]) / 3600.0
    if len(states) >= 2 and span_h > 0:
        lam = (len(states) - 1) / span_h
    else:
        lam = 1.0 / PRIOR_REFRESH_GAP_HOURS
    if len(states) < 2:
        return n_last, lam
    cutoff = last["t"] - 24.0 * 3600.0
    recent = [s for s in states if s["t"] >= cutoff]
    use = recent if len(recent) >= 3 else states
    xs = [math.log1p(hours_since_release(s["t"])) for s in use]
    ys = [math.log(max(1, s["n"])) for s in use]
    a, b = fit_log_log(xs, ys)
    if b < 0:
        b = 0.0
    n_target = math.exp(a + b * math.log1p(hours_since_release(target_epoch)))
    n_target = clamp(n_target, n_last, n_last * N_GROWTH_CAP)
    return n_target, lam


def tier_means(values, tiers):
    """Mean pooled value per tier (for shrinkage targets)."""
    sums = {}
    counts = {}
    for v, tier in zip(values, tiers):
        if v is None:
            continue
        sums[tier] = sums.get(tier, 0.0) + v
        counts[tier] = counts.get(tier, 0) + 1
    out = {}
    for tier in sums:
        out[tier] = sums[tier] / counts[tier]
    if out:
        out["__global__"] = sum(sums.values()) / sum(counts.values())
    return out


def solve_two_param(rows):
    """Weighted least squares for y = a*u + b*v (no intercept).

    rows are (u, v, y, w). Returns (a, b), each clamped at 0, or None when
    the design is degenerate.
    """
    suu = svv = suv = suy = svy = 0.0
    for u, v, y, w in rows:
        suu += w * u * u
        svv += w * v * v
        suv += w * u * v
        suy += w * u * y
        svy += w * v * y
    det = suu * svv - suv * suv
    scale = (suu + svv) or 1.0
    if abs(det) <= 1e-12 * scale * scale:
        return None
    a = (suy * svv - svy * suv) / det
    b = (svy * suu - suy * suv) / det
    return max(0.0, a), max(0.0, b)


def estimate_params(states, trophies, last_rates):
    """Per-trophy (alpha, beta) for the holder-count growth model.

    At each refresh j: dE = alpha * dN + beta * dt_hours * N_prev.
    Fitted per trophy by recency-weighted least squares over the history's
    refreshes (two parameters once there are at least 3 refreshes, alpha
    alone otherwise), then shrunk toward the tier mean. Falls back to the
    fresh-cohort prior (beta 0) when the history holds no refresh.
    Returns (alphas, betas), one pair per trophy.
    """
    n_troph = len(trophies)
    tiers = [t.get("tier", "bronze") for t in trophies]
    m = len(states)
    raw_a = [None] * n_troph
    raw_b = [None] * n_troph
    weights = [0.0] * n_troph
    if m >= 2:
        two_param = (m - 1) >= 3
        for i in range(n_troph):
            rows = []
            w_total = 0.0
            for j in range(1, m):
                dn = states[j]["n"] - states[j - 1]["n"]
                dt = (states[j]["t"] - states[j - 1]["t"]) / 3600.0
                c0 = states[j - 1]["counts"][i]
                c1 = states[j]["counts"][i]
                if c0 is None or c1 is None or dn <= 0:
                    continue
                w = dn * (RECENCY_DECAY ** (m - 1 - j))
                w_total += w
                rows.append((dn, dt * states[j - 1]["n"],
                             float(c1 - c0), w))
            weights[i] = w_total
            if not rows:
                continue
            if two_param:
                fit = solve_two_param(rows)
                if fit is not None:
                    raw_a[i], raw_b[i] = fit
                    continue
            # Alpha alone: weighted mean of dE/dN.
            num = sum(w * y / u for u, v, y, w in rows if u > 0)
            den = sum(w for u, v, y, w in rows if u > 0)
            if den > 0:
                raw_a[i] = num / den
                raw_b[i] = 0.0
    mean_a = tier_means(raw_a, tiers)
    mean_b = tier_means(raw_b, tiers)
    global_a = mean_a.get("__global__")
    global_b = mean_b.get("__global__")
    alphas = []
    betas = []
    for i in range(n_troph):
        if raw_a[i] is not None:
            target_a = mean_a.get(tiers[i], global_a)
            target_b = mean_b.get(tiers[i], global_b)
            target_a = 0.0 if target_a is None else target_a
            target_b = 0.0 if target_b is None else target_b
            s = weights[i] / (weights[i] + SHRINK_KAPPA)
            alphas.append(s * raw_a[i] + (1.0 - s) * target_a)
            betas.append(s * (raw_b[i] or 0.0) + (1.0 - s) * target_b)
        else:
            r = last_rates[i]
            if r is not None and r >= 99.99:
                alphas.append(1.0)
            elif r is None:
                alphas.append(0.0)
            else:
                alphas.append(PRIOR_FRESH_FACTOR * (r / 100.0))
            betas.append(0.0)
    return [max(0.0, a) for a in alphas], [max(0.0, b) for b in betas]


def predict_counts(states, trophies, last_rates, n_target, t_target):
    """Predicted holder count per trophy at (n_target, t_target)."""
    alphas, betas = estimate_params(states, trophies, last_rates)
    last = states[-1]
    dn = n_target - last["n"]
    dt = max(0.0, (t_target - last["t"]) / 3600.0)
    out = []
    for i in range(len(trophies)):
        c_last = last["counts"][i]
        if c_last is None:
            c_last = int(round(last_rates[i] / 100.0 * last["n"]))
        e = c_last + alphas[i] * dn + betas[i] * dt * last["n"]
        out.append(clamp(e, c_last, n_target))
    return out


def loo_spread(states, trophies, last_rates):
    """Pooled RMS logit error from predicting each refresh from earlier ones.

    Returns None when the history holds too few refreshes to learn from.
    """
    if len(states) < 3:
        return None
    errors = []
    for j in range(2, len(states)):
        sub = states[:j]
        e_pred = predict_counts(sub, trophies, last_rates,
                                states[j]["n"], states[j]["t"])
        n_target = states[j]["n"]
        for i in range(len(trophies)):
            c_act = states[j]["counts"][i]
            if c_act is None:
                continue
            r_pred = clamp(e_pred[i] / n_target * 100.0, 0.01, 99.99)
            r_act = clamp(c_act / n_target * 100.0, 0.01, 99.99)
            errors.append(logit_pct(r_act) - logit_pct(r_pred))
    if len(errors) < 10:
        return None
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def mixture_quantile(level, point_logit, weight_point, mu, sigma):
    """Invert the mixture CDF at `level` by bisection.

    Mixture: point mass at point_logit with weight weight_point, and a
    Normal(mu, sigma) with the remaining weight.
    """
    if sigma <= 0:
        sigma = 0.1
    lo = min(point_logit, mu) - 10.0 * sigma - 1.0
    hi = max(point_logit, mu) + 10.0 * sigma + 1.0

    def cdf(x):
        c = weight_point * (1.0 if x >= point_logit else 0.0)
        c += (1.0 - weight_point) * norm_cdf((x - mu) / sigma)
        return c

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_case(case):
    """Forecast one case. Returns the list of {"id", "q"} rows."""
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_min = case.get("horizon_minutes") or 0
    target_epoch = parse_time(case["target"])
    at_epoch = parse_time(case["at"])
    horizon_h = max(0.0, (target_epoch - at_epoch) / 3600.0)
    if horizon_min:
        horizon_h = horizon_min / 60.0

    # Last non-null rate per trophy, and last read overall.
    last_rates = [None] * len(trophies)
    for read in history:
        rates = read.get("rates") or []
        for i, r in enumerate(rates):
            if i < len(trophies) and r is not None:
                last_rates[i] = r
    for i, r in enumerate(last_rates):
        if r is None:
            last_rates[i] = 1.0

    states = distinct_states(history)
    if not states:
        # No history at all: wide spread around a flat 1% median.
        rows = []
        for t in trophies:
            qs = [inv_logit_pct(logit_pct(1.0) + z * 1.2) for z in Z_SCORES]
            rows.append({"id": t["id"], "q": [round(q, 4) for q in qs]})
        return rows

    last_n = states[-1]["n"]
    sigma_loo = loo_spread(states, trophies, last_rates)

    if len(states) >= 2:
        n_target, lam = forecast_players(states, target_epoch)
    else:
        lam = 1.0 / PRIOR_REFRESH_GAP_HOURS
        n_target = last_n * PRIOR_JUMP

    p_refresh = 1.0 - math.exp(-lam * horizon_h)
    p_refresh = clamp(p_refresh, 0.0, 1.0)

    if sigma_loo is not None:
        n_ratio = math.log(max(1.0, n_target / max(1, last_n)))
        sigma = math.sqrt(sigma_loo ** 2 + (0.30 * n_ratio) ** 2 + 0.04)
        sigma = clamp(sigma, SPREAD_MIN, SPREAD_MAX)
    else:
        sigma = PRIOR_SPREAD

    # Refresh-component centre from the cohort model (alpha + beta).
    e_target = predict_counts(states, trophies, last_rates,
                              n_target, target_epoch)

    rows = []
    for i, t in enumerate(trophies):
        r_last = last_rates[i]
        if r_last >= 99.99:
            # Universal trophy: everyone holds it, every quantile 100.
            rows.append({"id": t["id"], "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
            continue
        r_med = clamp(e_target[i] / n_target * 100.0, 0.01, 99.99)
        mu = logit_pct(r_med)
        point = logit_pct(r_last)
        qs = []
        for level in LEVELS:
            q_logit = mixture_quantile(level, point, 1.0 - p_refresh, mu, sigma)
            qs.append(round(inv_logit_pct(q_logit), 4))
        # Enforce non-decreasing after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": t["id"], "q": qs})
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    case_files = sorted(
        f for f in os.listdir(in_dir) if f.endswith(".in")
    )
    if not case_files:
        sys.stderr.write("no .in case files in %s\n" % in_dir)
        return 1
    for name in case_files:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        out = {"trophies": rows}
        out_name = name[:-3] + ".out" if name.endswith(".in") else name + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(out, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
