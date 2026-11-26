# ADR-0014: PSI with baseline quantile bins, epsilon 1e-4 and a 500-row minimum

- Status: Accepted
- Date: 2026-11-14

## Context

The drift alerts use the industry PSI thresholds 0.1/0.25, which do not fix the method. PSI is unstable with empty bins and small samples.

## Decision

10 quantile bins from the baseline with open outer edges; one bin per category plus `__other__`; epsilon 1e-4 with renormalisation; no PSI is pushed below `drift.min_samples` (500). Sources: Siddiqi, *Credit Risk Scorecards* (Wiley, 2006) for the thresholds; Yurdakul, *Statistical Properties of Population Stability Index* (2018), https://scholarworks.wmich.edu/dissertations/3208, for the sample-size and binning effects.

## Consequences

- Thresholds mean the same thing across features and models.
- Small traffic produces "skipped" results instead of false alerts.
