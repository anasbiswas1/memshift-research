# Plan v2, amendment 3

Md Anas Biswas | 18 November 2026 | Applies to plan v2 with amendments 1 and 2 | Frozen before notebook 09 is run

## 1. Why

Three gaps in the method as confirmed on users 160 to 189 (notebook 08), each visible in the committed reports.

**Held-out risk comes from units the method never sees.** Under Mistral (M7) the strictest labelled candidate K_1 leaves 1.67% of the 600 held-out tasks attacked (10 hits; `reports/08_policy_outcomes.csv`, `r_heldout` of `M7|RLMA|B200|g1`). Every one of the 10 hits, traced in `reports/01_records/mistral_7b/`, is a (record, category) unit with zero exposure on the log, while the same record has another unit on the log scored 0.64 to 0.98 by the labelled score, above every threshold (λ_5 = 0.55). The record was quarantined in the category where the log retrieved it and left alone in the category where only new phrasings retrieve it; 8 of the 10 are calendar queries. Nine users have held-out hits under K_1, one at 10% of its 20 held-out tasks. The strictest candidate cannot fix this, because units with zero log exposure are not units of the method at all.

**The score is trained on the new reader.** The score of a migration is fitted on probe and end-to-end labels of the new reader on the development users (Section 4.1 of the manuscript; 2,780 probes and 6,000 task episodes for Mistral). An operator moving to a model that has never served the development service does not have those labels. On the 1,912 units that M1 and M7 share (same survivor store), the Llama-trained and the Mistral-trained labelled scores have Spearman correlation 0.89; their candidates at matched benign-exposure shares overlap with Jaccard 0.97, 0.92, 0.82, 0.74 and 0.55 for K_1 to K_5 (`reports/08_units.csv.gz`, columns `s` of M1 and M7). A score fitted on other readers' labels may be good enough.

**The certificate is aggregate; the residual risk is not.** On the log, M7's loosest candidate K_5 leaves 6 attacked tasks, all payments (2.0% of payment tasks, 0 in the other categories); M1's K_5 leaves one user at 2.5% of its tasks and M7's K_5 one at 7.5% (`reports/08_candidate_outcomes.csv.gz`). The certificate of Proposition 1 is for the whole log; a reader of the paper should see the distribution behind it.

## 2. Changes

**2.1 Zero-exposure extension (part of the method).** Let *E* be the set of units with positive exposure on the log under the new retriever (the units of Section 3.4). For a candidate *K* of either tier,

K^ext = K ∪ { (i, c) : (i, c) ∉ E and (i, c′) ∈ K for some category c′ }.

A record that the candidate quarantines in some category is quarantined as well in every category in which the log never retrieved it. Records quarantined whole by the census already cover every category. Units in *E* that the candidate keeps are not touched: a record kept in an exposed category is a deliberate choice at that risk level, and the audit measures it.

Property. For every logged task the context under K^ext equals the context under K, because the added units are never retrieved on the log. Hence y_{P1,K^ext}(q) = y_{P1,K}(q) for every q in Q, r_Q is unchanged, benign task success on the log is unchanged, and every audit outcome, certificate and replay of amendment 2 holds for K^ext exactly as for K. The extension changes only what happens on tasks outside the log. Its cost on new tasks is the benign records it withholds there; its benefit is the poisoned records it withholds there. Both are measured on the held-out tasks.

**2.2 Transferred score (an evaluation of the method, not a change to it).** For a migration whose new reader is L_1, the transferred scores s^T and s′^T are fitted with the recipe of amendment 1 (same features, same gradient-boosted trees, same user-grouped folds, same seed) on the pooled training units of the migrations whose new reader is not L_1: for M1 (new reader Llama-3.1-8B) the training units of M2, M3 and M7 (labels under Qwen2.5-7B, Qwen3-8B and Mistral-7B); for M2 those of M1, M3 and M7; for M7 those of M1, M2 and M3 (`reports/05_training_units.csv.gz`, `reports/07_training_units.csv.gz`). Thresholds come from the pooled out-of-fold scores at the grids of amendment 1. The new reader contributes no label. The two-tier method is then run with s^T and s′^T in place of s and s′: its own censuses (200 units by w·s^T, and by w·s′^T when the fallback tier is entered), its own candidates, the same audit.

**2.3 Residual-risk breakdown (reporting).** For every candidate of both tiers and both scores, the attacked-task rate by task category and by user, on the log and on the held-out tasks, from the saved per-task outcomes.

**Unchanged.** The two-tier audit, its levels and sample sizes, the censuses, the serving rule, the baselines, the inclusion rule, the laundering design and claims C5 to C7. Laundered labels are not re-run with the transferred score.

## 3. Confirmation: notebook 09

Users 160 to 189, migrations M1, M2 and M7, the recorded episodes of notebook 08 reused wherever a context is unchanged. For each migration and each score (own, transferred): the labelled and the label-free candidates, their log outcomes, and the held-out outcomes of the extended candidates; for the own score also the held-out outcomes of the unextended candidates (notebook 08). The two-tier replay of amendment 2 with the same per-cell seeds as notebook 08 (the own-score replay must reproduce `reports/08_two_tier_replay.csv` exactly, which checks the reconstruction).

Claims, every cell reported (a cell is a migration and a risk limit, clean labels):

- **C8 (extension).** (a) In every cell, the mean held-out attacked-task rate of the policy the two-tier audit deploys is no higher with the extension than without it. (b) With the extension it is at most α in every cell. (c) The held-out benign task success of the deployed policy falls by at most 1.0 point with the extension in every cell.
- **C9 (transfer).** With the transferred scores, (a) the two-tier audit certifies a candidate in at least 95% of the replays in every cell, and (b) its certified task success is at least the best amendment 1 comparator's (`reports/08_audit_replay.csv`) in at least two thirds of the cells. The gap to the own-reader score is reported, not claimed.
- The residual-risk breakdown is reported; no claim.

The diagnosis above is what the extension was designed on; the held-out numbers of notebook 09 are the ones reported.
