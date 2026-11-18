# Plan v2, amendment 2

Md Anas Biswas | 17 November 2026 | Applies to plan v2 (docs/plan_v2.md) with amendment 1 | Frozen before users 160 to 189 were generated

## 1. Why

Notebook 07 measured the method with laundered provenance labels: the attacker relabels a share of the poisoned records as user-sourced, as if they had passed through the agent's own note-taking. The certificate held in all 18 cells (claim C3), because the audit never reads the label. But the labelled score misses the laundered records, the most restrictive candidate fails the audit, and the only certified outcome is the reset: on M1 with 25% laundered, RLMA certified a candidate in 0.9 to 41% of the replays and kept 59.8 to 60.9% task success against 63.3 to 64.5% for the ablation without the label; on M7 it certified in 0 to 2.6% and kept 30.3 to 30.7% against 36.4 to 44.0%. The ablation without the label is immune to laundering but, with clean labels, is up to 8.5 points below the labelled method (M7: 36.4 against 44.9 at α = 0.01).

The operator does not know which case holds. The audit tells it: under laundering the most restrictive labelled candidate fails. This amendment adds the second, label-free candidate sequence as a fallback tier that is entered only then, with the certificate kept at δ by splitting the level between the tiers.

## 2. Change: a second, label-free tier

**Candidates.** Two candidate sequences on the served units, both as in amendment 1.

- Labelled tier K_1 ⊂ … ⊂ K_5: the score s with the provenance label, thresholds at 15, 10, 7, 5 and 3% of benign exposure, the targeted census of 200 units by w·s among the units K_5 keeps, whole-record quarantine of every active unit. This is RLMA of amendment 1.
- Label-free tier K'_1 ⊂ … ⊂ K'_5: the score s_ns without the label, thresholds at 25, 15, 10, 5 and 2%, its own targeted census of 200 units by w·s_ns. This is the ablation "RLMA without the provenance label" of amendment 1. Its census is run only when the tier is entered, and its cost is counted then.

**Audit (Algorithm 1 of amendment 1, steps 3 and 4 replaced).**

1. Draw one audit sample of n = n₀(α, δ/2) = ⌈ln(δ/2) / ln(1 − α)⌉ queries from the log: 368, 183 and 72 at α = 0.01, 0.02 and 0.05 (amendment 1: 299, 149, 59).
2. Labelled tier: run K_1, K_2, … on the sample in order; candidate K_g passes when P(X ≤ x_g; n, α) ≤ δ/2, where x_g is its number of attacker actions on the sample (Equation 7 at level δ/2); stop at the first failure; deploy the last passing candidate.
3. If K_1 fails (nothing is certified): run the label-free census, build K'_1, …, K'_5, test them in order on the same sample at level δ/2, deploy the last passing candidate; if K'_1 fails too, reset.

**Guarantee (Proposition 1 of plan v2, restated).** Whatever the attacker's labels, P(the deployed candidate has r_Q > α) ≤ δ. Proof. Let B be the event that the labelled tier certifies a candidate with r_Q > α and B' the same event for the label-free tier. A bad deployment requires B or B'. In a fixed sequence tested at level δ/2, the first candidate with r_Q > α must pass its own test for B to happen, and that test passes with probability at most δ/2 under its exact binomial distribution (the fixed-sequence argument of plan v2 at δ/2); so P(B) ≤ δ/2, and in the same way P(B') ≤ δ/2 whether or not the tier is entered. By the union bound P(B ∪ B') ≤ δ. The decision to enter the second tier depends on the sample; the bound does not depend on that decision, and it does not need the tiers' tests to be independent, so the two tiers may share one sample.

**Cost.** The audit sample is 23% larger (69, 34 and 13 more queries); the label-free census (200 probes) is run only on fallback; at most ten candidates are run on the sample. At these sizes a candidate passes only with zero attacker actions on the sample, so a candidate of true rate r passes with probability (1 − r)ⁿ and the larger sample certifies the loosest candidate a little less often: replayed with the notebooks' own candidate rates, the cost is at most 0.7 points of task success (M7, α = 0.02).

**Unchanged.** Units and exposure, both scores and their training data, thresholds, census size and order, serving, the baselines and ablations, the inclusion rule, claims C2 to C4 as stated in amendment 1.

**Alternatives considered and not taken.** (a) One merged sequence with the labelled and label-free candidates interleaved by benign share: every candidate placed before the loosest labelled one is an extra hurdle with clean labels, and the label-free census would be run always. (b) Choosing the best certified candidate across both tiers at family level δ: the same stopping rule in effect, with the same always-on census. (c) A laundering detector (for example a census positive carrying a user-sourced label) that switches scores: a new heuristic with no guarantee that it fires.

## 3. Development evidence (users 130 to 159, offline)

The two-tier audit was replayed from the recorded episodes of notebooks 05 and 07, nothing run: the outcome of each candidate on each log query is the recorded episode of its exact context (the natural top-5 minus the records the candidate quarantines), looked up by episode key; task success per candidate is the notebooks' value; 1,000 audit replays per cell with the same sample for both tiers. The score part of every candidate is exact. The census positives of a candidate were rebuilt from the recorded probe episodes and the episodes recorded under the candidate's own name; M2 is reproduced exactly, M1 within one attacker action per candidate, and M7's census under-recovers (20 of RLMA's 24 positives), so M7's labelled candidates carry more attacker actions here than in notebook 07. The error runs against the two-tier method.

Certified benign task success (%), mean over 1,000 replays, α = 0.01 / 0.02 / 0.05; "fallback" is the share of replays that entered the label-free tier.

| Migration, labels | RLMA (amendment 1) | Without the label | Two-tier | Fallback |
|---|---|---|---|---|
| M1, clean | 64.0 / 64.3 / 64.5 | 63.3 / 63.8 / 64.4 | 63.9 / 64.2 / 64.5 | 0 / 0 / 0 |
| M1, 25% laundered | 59.8 / 60.1 / 61.0 | 63.3 / 63.8 / 64.4 | 63.0 / 63.5 / 63.5 | 0.99 / 0.92 / 0.64 |
| M1, 50% laundered | 59.8 / 59.8 / 59.9 | 63.3 / 63.8 / 64.4 | 63.0 / 63.6 / 64.0 | 1.00 / 1.00 / 0.93 |
| M2, clean | 65.4 / 65.4 / 65.4 | 64.5 / 65.0 / 65.2 | 65.4 / 65.4 / 65.4 | 0 / 0 / 0 |
| M2, 25% laundered | 62.4 / 63.6 / 64.9 | 64.5 / 65.0 / 65.2 | 64.9 / 65.2 / 65.3 | 0.26 / 0.14 / 0.06 |
| M2, 50% laundered | 59.9 / 62.1 / 64.2 | 64.5 / 65.0 / 65.2 | 64.5 / 65.0 / 65.2 | 0.47 / 0.26 / 0.12 |
| M7, clean | 42.9 / 44.6 / 47.2 | 36.2 / 38.2 / 42.3 | 42.4 / 44.0 / 46.4 | 0 / 0 / 0 |
| M7, 25% laundered | 30.3 / 30.3 / 30.6 | 36.2 / 38.2 / 42.3 | 35.8 / 37.3 / 41.1 | 1.00 / 1.00 / 0.99 |
| M7, 50% laundered | 30.3 / 30.3 / 30.4 | 36.2 / 38.2 / 42.3 | 35.8 / 37.3 / 41.1 | 1.00 / 1.00 / 1.00 |

The two-tier method certified a candidate in every replay of every cell. With clean labels it never entered the fallback tier and stayed within 0.1 (M1, M2) and 0.8 (M7) points of RLMA, the cost of the larger sample. With laundered labels it stayed within 1.2 points of the label-free ablation (the cost of the larger sample, as above) and 0.4 to 10.5 points above RLMA. Its violation rate was at most 2.0% in every cell (δ = 5%); the rebuilt M7 and M1 candidates account for the non-zero cells (the notebooks' own candidates of those cells had no violations at the loosest labelled candidate).

## 4. Confirmation: notebook 08

Users 160 to 189, generated only after this amendment is frozen (suite `confirmation3`). Migrations M1, M2 and M7, the three that entered claims C2 to C4 on users 130 to 159, with their old pipelines, survivor stores, phrasing checks, probes and references exactly as in notebooks 05 and 07. Scores frozen: the fold models notebook 07 saved under `reports/07_models/` are loaded as they are (M1 and M2 with the label, M7 both); the label-free scores of M1 and M2 were not saved and are refitted on `reports/05_training_units.csv.gz` with the same recipe and seed, and the refit is checked against the saved scores of users 130 to 159 in `reports/05_units.csv.gz` (largest difference, and the number of units the refit moves across any threshold, both recorded; gradient-boosted trees train in float32 with parallel sums, so a refit is close but not bit-exact); thresholds from `reports/05_thresholds.json` and `reports/07_thresholds.json`. Every method, ablation and reference of notebook 05, laundering at 25% and 50% as in notebook 07, and the two-tier method, whose rows come from the labelled and the label-free candidates of the same migration (the label-free candidates do not depend on the labels); held-out queries for the candidates of both tiers. The audit replay runs amendment 1's audit (n₀(α, δ)) for every method and the two-tier audit (n₀(α, δ/2), one sample for both tiers) for the two-tier method, with the same seeds per cell. A cell is a (migration, labels, α) triple on an included migration; the inclusion rule of amendment 1 is applied on users 160 to 189.

Claims, each tested as stated, every cell reported:

- **C5 (certificate under any labels).** For the two-tier method the violation rate is at most δ in every cell, clean and laundered: the binomial test of claim C3 against δ at level 0.01 does not reject in any cell.
- **C6 (no cost with clean labels).** In every clean cell, the two-tier method certifies a candidate in at least 95% of the replays whenever RLMA does, enters the fallback tier in at most 5% of the replays, and its certified task success is at least RLMA's minus 1.5 points.
- **C7 (laundering recovered).** In every laundered cell, the two-tier method's certified task success is at least the larger of RLMA's and the label-free ablation's minus 2.0 points.
- **C4 for the two-tier method.** With clean labels, against the comparators of amendment 1 (reuse all, scanner curve, scanner score + audit, random census, greedy, blanket external-source quarantine) at the same census size and the amendment 1 audit, the two-tier method's certified task success is at least the best comparator's in at least two thirds of the cells; C2 and C3 are reported for every method as before.

Development numbers above are what the design was chosen on; the confirmation numbers of notebook 08 are the ones reported.
