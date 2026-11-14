Research and implementation plan, version 2

Md Anas Biswas \| 13 November 2026 \| Supersedes plan v1 of the same date \| Repository: memshift-research

# 1. Summary

An LLM agent keeps a long-term memory that may hold records planted by an attacker. The current model ignores them, so the system looks safe. When the operator upgrades the model or the embedding model that retrieves memory, and keeps the same store, some of those records can start to work.

This plan builds a revalidation procedure the operator runs at migration time. A cheap score and a small number of record checks build a few candidate quarantine policies. Each candidate is frozen and run end to end on randomly sampled real queries, from the most to the least restrictive, and the procedure deploys the last candidate that passes before the first one that fails. The certificate says: with probability at least 1 − δ, the share of queries on which the deployed, quarantined system sends data or money to an attacker is at most α.

Version 2 answers an external review of plan v1. Its central change: the certificate is now about the quarantined system as it actually runs, not about labels measured on records before quarantine. A record's behaviour depends on what else is in the context, and removing other records changes that context, so only executing the frozen policy can certify it.

**The score buys utility; the audit buys the guarantee.** That split from v1 is kept. The statistics of the audit are standard (an exact binomial acceptance test in a fixed sequence). The paper's contribution is the migration measurement and a policy construction that keeps more legitimate memory at the same certified risk, shown against fair baselines and ablations.

## 1.1 What changed from plan v1

| **Review item**                                             | **Plan v2**                                                                                                                                                                                                                       |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Two-probe labels do not measure deployment risk (§2)        | Unit labels stay, with a natural or forced flag, for the score, the census and the migration measurement only. The certified quantity is the end-to-end unauthorised-action rate of the frozen policy (Equation 4).               |
| Quarantine changes the certified system (§3)                | Quarantine is executed: removed records are filtered before ranking and the next records fill the context. Every candidate policy is run end to end; no outcome is copied between policies.                                       |
| Preferred repair: audit the exact quarantined system (§4)   | Adopted as the method (Section 6).                                                                                                                                                                                                |
| Audit can keep a detected positive (§5)                     | Certification requires zero observed unauthorised actions in the audit (n = n₀), so a policy with an observed attack is never certified.                                                                                          |
| Audit size 473, not 300 (§5)                                | With certification at zero observed positives, n₀ = 299, 149, 59 at α = 0.01, 0.02, 0.05 and δ = 0.05 (exact binomial, checked).                                                                                                  |
| Boundary cases (§5)                                         | Explicit rules in Algorithm 1: insufficient budget, all candidates failing, empty populations.                                                                                                                                    |
| Old pipeline measured with new contexts (§6)                | All four pipelines (R₀,L₀), (R₀,L₁), (R₁,L₀), (R₁,L₁) are run; Equation 5 separates retrieval, reader and interaction effects.                                                                                                    |
| Survivor store (§6)                                         | Built from observed attacks of the actual old pipeline on the query log, then re-verified once; the full transition table is reported.                                                                                            |
| Seeded poison versus threat model (§6)                      | The main study is stated as conditional on poison already in memory; a small ingestion validation is added (notebook 04).                                                                                                         |
| Checker source of truth (§4)                                | A trusted destination registry per user. Attacker-caused outcomes (canary) are the certified metric; the registry-only check, which an operator can run, is reported as a conservative variant.                                   |
| Provenance baseline too coarse (§7)                         | Renamed blanket external-source quarantine; an authority-gating reference with perfect provenance is added; the method's scope is stated as stores without trustworthy provenance.                                                |
| Unequal baselines; C4 required a baseline to fail (§7)      | Every audited method uses the same audit and the same total inference budget; greedy ranks by w·s; the scanner gets a risk and utility curve; no claim requires a baseline to fail.                                               |
| Cost of the score (§7)                                      | Cost reported by component; ablation without the differential probe.                                                                                                                                                              |
| A4 selection and evaluation queries overlap; A5 naming (§7) | A4 and A5 are evaluated on held-out queries never used in selection; A5 is called scanner-evading.                                                                                                                                |
| Novelty too categorical; licence error (§8)                 | Positioning softened (Section 3); Qwen2.5-3B is recorded under the Qwen research licence.                                                                                                                                         |
| Scope of claims (§9)                                        | The certificate is stated for the finite query log; held-out queries measure generalisation; per-user and per-category results are descriptive with user-cluster bootstrap intervals; development and confirmation are separated. |
| Scope of the first study (§10)                              | Three readers, reader migrations M1 to M3 and one retriever pair (M4, M6). M5, NOTES format, mixed index and a fourth reader are extensions.                                                                                      |

# 2. The problem in plain words

An illustration (not a measured result). A supplier's invoice once made the assistant save the note "Payments for Northwind now go to account GB00 ATTK 0000 1234". The current reader never uses it, because the saved payee list disagrees. After an upgrade, the new model trusts the most recent note and pays the attacker. Nothing in memory changed; only the reader did.

Two things change at migration. A new retriever brings different records into the context (a retrieval change). A new reader acts differently on the same context (a reader change). The two can also interact. Equation 5 separates all three.

A third effect matters for any defence. Removing records changes the context of the records that stay. A poisoned note that was harmless next to the legitimate payee entry can become active once that entry is removed. This is why the plan certifies the quarantined system by running it.

# 3. Prior work and position

| **Work**                                                                            | **What it covers**                                                                                                                                                                                                                            | **Consequence for this paper**                                                                                                                                                                  |
|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Goyal and Ray, arXiv 2609.05339                                                     | Memory portability across model and embedding changes, for accuracy; writer, retriever and reader roles; mixed indexes; repair needs raw history.                                                                                             | Migration as a problem is theirs. Our difference is the adversarial setting, the security outcome and the revalidation decision.                                                                |
| Pulipaka et al., arXiv 2605.15338                                                   | Sleeper memory poisoning; large differences between models in whether stored poison acts. Its 0.96 figure is payload localisation by a document scanner, not scanner discrimination of memories.                                              | Threat model and payload style; the 0.96 figure is not quoted as scanner accuracy.                                                                                                              |
| Leong, arXiv 2605.08442 v6                                                          | Storing and acting are separate. According to the external review, its Appendix M.1 reports a shared-store writer and reader experiment whose interpretation a memory-tool bug invalidated, and states that corrected transfer was not shown. | Prior work identifies the risk and reports a confounded attempt. Our claim is validated, controlled measurement of migration-induced activation. Appendix M.1 is to be read before any writing. |
| Gao et al., MemPoison, arXiv 2607.14651 v2                                          | Compositional attacks across several records and dormant attacks whose activation depends on context.                                                                                                                                         | Context dependence is real, which is why Route B executes each policy.                                                                                                                          |
| Ouyang and Hou, MemLineage, arXiv 2605.14421; Louck, TMA-NM, arXiv 2606.24322       | Provenance and authority gating that keeps recall while refusing sensitive actions justified by external ancestry.                                                                                                                            | When trustworthy provenance exists, authority gating is the right tool; it is a reference in our evaluation. Our method targets stores without trustworthy provenance.                          |
| Shekhar et al., arXiv 2305.06884 (Appendix C)                                       | Risk-limiting audits under a budget, including the misstated fraction remaining after audited items are corrected.                                                                                                                            | Residual-risk auditing is not new; the audit statistics are cited as foundations.                                                                                                               |
| Angelopoulos et al., Learn then Test, arXiv 2110.01052                              | Fixed-sequence testing for risk control; validity does not need monotonicity.                                                                                                                                                                 | Used for the candidate sequence.                                                                                                                                                                |
| Opoku and Banahene, ToolChain-CRC, arXiv 2606.18467; Sharma, SMSR, arXiv 2606.12703 | Risk control for agent trajectories under drift; a per-query certified memory defence.                                                                                                                                                        | Different objects (no migration, no store-level revalidation); cited and contrasted.                                                                                                            |

**Open, as far as the checked literature shows:** a validated measurement of how reader and retriever migration activates stored poison in an agent, and a budgeted migration-time procedure whose certificate covers the quarantined system as deployed. **Checks still to do before writing:** read Leong v6 Appendix M.1 and Shekhar et al. Appendix C in full; search OpenReview (ICLR 2027 submissions, NeurIPS 2026 workshops); confirm the licence of the sleeper-poisoning repository before reusing payload templates.

# 4. Contribution and the objections it must survive

**Draft contribution statement.** A migration-time revalidation procedure for agent memory without trustworthy provenance. A cheap score and a small census build candidate quarantine policies per task category; each candidate is executed end to end on sampled real queries, and the least restrictive candidate with no observed unauthorised action is certified: with probability at least 1 − δ, its rate of attacker-caused actions on the query log is at most α. It comes with a controlled measurement of how reader and retriever migration activate stored poison, separated into reader, retrieval and interaction effects.

**Honest scope.** The audit is a standard acceptance test; its validity for any construction is inherited, not new. The paper is judged on (a) the migration measurement and (b) how much legitimate memory the method keeps at a certified risk compared with fair baselines at the same inference budget.

| **Expected objection**                                   | **How the plan answers it**                                                                                                                                                                                                                                                                |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| "This is just testing the new system before deployment." | Partly true, and stated. The contribution is the construction of policies that pass the test while keeping utility, the cost accounting, and the migration measurement. The baselines include testing alone (reuse all, then audit) and random or greedy construction with the same audit. |
| "Provenance solves it."                                  | Authority gating with perfect provenance is reported as a reference. The method is for legacy stores and agent-derived entries without trustworthy provenance; lack of applicability is argued separately from performance.                                                                |
| "A one-pass scan solves it."                             | Claim C2 on end-to-end risk, with the scanner's full risk and utility curve.                                                                                                                                                                                                               |
| "The certificate is for one query log."                  | Stated as a finite-log certificate; held-out queries per user measure generalisation and are reported separately.                                                                                                                                                                          |
| "The checker assumes ground truth."                      | The certified metric uses the experiment's canaries. The registry-only check, which an operator can run, is reported as a conservative variant with its utility cost.                                                                                                                      |

# 5. Threat model

**Attacker.** Has already caused text to be written into memory through ordinary channels (documents, web pages, emails), as in sleeper memory poisoning. The main study is conditional on that: poisoned records are seeded into the store. A small ingestion validation (notebook 04) checks that the old agent itself writes comparable records and that later attacks act through memory alone, with the source document removed. The attacker knows the current reader and may know the likely next one, and may choose phrasings that evade a known scanner. The attacker cannot change the operator's query traffic, see the audit's random draws, or influence the checker.

**Operator.** Has the store, the old and new pipelines, a log of recent queries, a budget of inference calls, and a trusted destination registry per user: the address book, the saved payees, and contacts and updates the user confirmed directly.

**Outcome.** An unauthorised action is a consequential call (send, invite, share, pay) whose destination is outside the trusted registry. In the experiment, attacker-caused actions are identified exactly by canaries planted with each payload; this is the certified metric. The registry-only indicator, which also counts model guesses and any memory-only destination, is computable by an operator and is reported as a conservative variant.

**Out of scope.** Poison written after migration; a compromised operator; attacks on the checker; queries outside the logged traffic (handled descriptively by held-out queries); answer-only steering with no consequential action.

# 6. Method

## 6.1 Objects

- A memory service holds records for many users. A **unit** u = (i, c) is record i used for task category c ∈ {email, payments, calendar, documents}. Quarantine is per unit.

- A pipeline P = (R, L) has a retriever R and a reader L. The old pipeline is P<sub>0</sub> = (R<sub>0</sub>, L<sub>0</sub>), the new one P<sub>1</sub> = (R<sub>1</sub>, L<sub>1</sub>). Q is the query log; each query has a category.

- A policy K is the set of quarantined units. Serving under K filters quarantined units before ranking, so the next records fill the top-k context: Top<sub>k</sub><sup>R,K</sup>(q) is the top k of the records not quarantined for category c(q).

## 6.2 Unit signals: exposure and probe labels

These feed the score, the census and the migration measurement. They are not the certified quantity.

w<sub>u</sub>(R) = (1 / \|Q\|) · \#{ q ∈ Q : c(q) = c and m<sub>i</sub> ∈ Top<sub>k</sub><sup>R</sup>(q) } (1)

a<sub>u</sub>(P) = 1\[ P acts for the attacker on at least one of the J probes of unit u \] (2)

A probe is natural when the retriever returned the record for that query, and forced when the record had to be placed in the context. The flag ν<sub>u</sub> = 1 when every probe of u is natural. Results are always reported split by ν<sub>u</sub>.

## 6.3 The certified quantity

y<sub>P,K</sub>(q) = 1\[ P under policy K takes an attacker-caused consequential action on q \] (3)

r<sub>Q</sub>(P, K) = (1 / \|Q\|) · Σ<sub>q ∈ Q</sub> y<sub>P,K</sub>(q) (4)

y is the outcome of one recorded greedy execution of the full episode, so r<sub>Q</sub> is a property of a finite population of recorded outcomes. Run-to-run variation of the serving stack is measured separately by the stability study.

## 6.4 Migration measurement

With the store unquarantined (K = ∅), the change in risk from the old to the new pipeline splits exactly into three parts:

r(R<sub>1</sub>,L<sub>1</sub>) − r(R<sub>0</sub>,L<sub>0</sub>) = \[r(R<sub>1</sub>,L<sub>0</sub>) − r(R<sub>0</sub>,L<sub>0</sub>)\] + \[r(R<sub>0</sub>,L<sub>1</sub>) − r(R<sub>0</sub>,L<sub>0</sub>)\] + I (5)

The first bracket is the retrieval effect under the old reader, the second the reader effect under old contexts, and I = r(R<sub>1</sub>,L<sub>1</sub>) − r(R<sub>1</sub>,L<sub>0</sub>) − r(R<sub>0</sub>,L<sub>1</sub>) + r(R<sub>0</sub>,L<sub>0</sub>) the interaction. For reader-only migrations (R<sub>0</sub> = R<sub>1</sub>) only the reader effect exists.

**Survivor store.** Remove every poisoned record that caused an attacker action under the actual old pipeline (R<sub>0</sub>, L<sub>0</sub>) on any query of the log; run the old pipeline once more on the remaining store and remove any record that now acts (at most two rounds, both reported). The result is observationally dormant under the old system. The full old-to-new transition table is reported before survivors are selected.

## 6.5 Policy construction (development side)

- **Score.** s<sub>u</sub> ∈ \[0,1\], a calibrated estimate that unit u acts under P<sub>1</sub>, from: scanner score, exposure change \|w<sub>u</sub>(R<sub>1</sub>) − w<sub>u</sub>(R<sub>0</sub>)\|, record features (names a contact, payee, address or link), and a light differential probe (one short generation from each reader). Fitted on calibration users only.

- **Census.** The B<sub>c</sub> units with the largest w<sub>u</sub>(R<sub>1</sub>) · s<sub>u</sub> are probed under P<sub>1</sub>; units found active are quarantined in every candidate.

- **Candidates.** K<sub>g</sub> = C<sup>+</sup> ∪ { u : s<sub>u</sub> ≥ λ<sub>g</sub> } for a fixed grid λ<sub>1</sub> \< λ<sub>2</sub> \< … \< λ<sub>G</sub>, ordered from most quarantine to least, with G = 5. The reset policy K<sub>reset</sub> (all memory withheld) is the fallback; with no memory in context no canary can be reached, so its attacker-caused risk is zero by construction.

## 6.6 The audit (certification side)

Draw n audit queries independently and uniformly from Q, with replacement, after all candidates are fixed. Run the candidates in order on the same audit queries, each with its own execution, and count unauthorised actions:

X<sub>g</sub> = Σ<sub>j=1..n</sub> y<sub>P1,Kg</sub>(q<sub>j</sub>), p<sub>g</sub> = Pr\[ Binomial(n, α) ≤ X<sub>g</sub> \] (6)

g\* = max{ g : p<sub>1</sub> ≤ δ, …, p<sub>g</sub> ≤ δ }; if p<sub>1</sub> \> δ, deploy K<sub>reset</sub> (7)

n = n<sub>0</sub>(α, δ) = ⌈ ln δ / ln(1 − α) ⌉ (8)

With n = n<sub>0</sub>, p<sub>g</sub> ≤ δ holds only when X<sub>g</sub> = 0, so a certified policy never has an observed unauthorised action in its audit. n<sub>0</sub> is 299, 149 and 59 at α = 0.01, 0.02 and 0.05 with δ = 0.05. Testing stops at the first failure, so at most G · n audited executions are needed, and fewer in practice.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<tbody>
<tr class="odd">
<td><p><strong>Algorithm 1. Migration-time revalidation with an end-to-end audit (working name RLMA)</strong></p>
<p>Input: store, new pipeline P<sub>1</sub>, query log Q, calibrated score s, census size B<sub>c</sub>, grid λ<sub>1</sub> &lt; … &lt; λ<sub>G</sub>, risk limit α, confidence δ.</p>
<p>1. Census: probe the B<sub>c</sub> units with the largest w<sub>u</sub>(R<sub>1</sub>) · s<sub>u</sub>; C<sup>+</sup> is the set found active.</p>
<p>2. Freeze the candidates K<sub>g</sub> = C<sup>+</sup> ∪ { u : s<sub>u</sub> ≥ λ<sub>g</sub> }, the retrieval rule (filter, then top-k), the model configuration and the tools.</p>
<p>3. Draw q<sub>1</sub>, …, q<sub>n</sub> uniformly from Q with replacement, n = n<sub>0</sub>(α, δ).</p>
<p>4. For g = 1, …, G: run P<sub>1</sub> under K<sub>g</sub> on every audit query; if any unauthorised action occurs (X<sub>g</sub> &gt; 0), stop.</p>
<p>5. Deploy K<sub>g*</sub>, the last candidate with X = 0. If K<sub>1</sub> already fails, deploy K<sub>reset</sub>. If the budget cannot pay for n executions of K<sub>1</sub>, report insufficient budget and deploy K<sub>reset</sub>.</p>
<p>Output: deployed policy, the certificate r<sub>Q</sub>(P<sub>1</sub>, K) ≤ α at confidence 1 − δ, every observed unauthorised action with the record that caused it, and the cost by component.</p></td>
</tr>
</tbody>
</table>

## 6.7 Guarantee

**Proposition 1.** If the candidates, their order and the serving configuration are fixed before the audit draws, the audit queries are independent and uniform on Q, and the checker observes y exactly, then Pr\[ r<sub>Q</sub>(P<sub>1</sub>, K<sub>deployed</sub>) \> α \] ≤ δ, whatever the score and the census did, including a score an attacker has gamed.

**Proof sketch.** For a fixed candidate, X<sub>g</sub> is Binomial(n, r<sub>Q</sub>(P<sub>1</sub>, K<sub>g</sub>)). If r<sub>Q</sub> \> α, the probability of X<sub>g</sub> = 0 is (1 − r<sub>Q</sub>)<sup>n</sup> \< (1 − α)<sup>n</sup> ≤ δ. Under fixed-sequence testing a wrong certificate requires the first candidate in the sequence with r<sub>Q</sub> \> α to pass, which has probability at most δ; monotonicity of r<sub>Q</sub> in g is not needed. K<sub>reset</sub> has r<sub>Q</sub> = 0. The full proof goes into the paper.

**Scope.** The certificate is for the finite log Q and is aggregate over users and categories. Per-user and per-category rates are reported descriptively, with intervals from a bootstrap that resamples users. Generalisation to new traffic is measured on held-out queries (Section 7.1).

# 7. Evaluation design

## 7.1 Testbed

- The synthetic memory service of notebook 00 (100 users, 60 records each, 5 poisoned; users 0 to 29 calibrate the score, 30 to 99 are audited), unchanged.

- **Held-out queries.** 20 more queries per user from the same templates with a different seed, stored separately. They are never used for scores, census, A4 selection or the audit; they measure whether a certified policy holds on new traffic, and they evaluate A4 and A5 on queries their selection never saw.

- **Trusted registry** per user: address book, saved payees, user-confirmed updates and memory-only contacts the user gave. The canaries are outside it by construction.

- **Confirmation users** 100 to 129 are generated only if a development change follows a failed gate; adapted settings are confirmed on them before they are reported as results.

## 7.2 Pipelines and migrations

| **ID**     | **Migration**                                                                 | **Pipelines run**     |
|------------|-------------------------------------------------------------------------------|-----------------------|
| M1         | Qwen2.5-7B to Llama-3.1-8B (bge-large-en-v1.5)                                | (R,L₀), (R,L₁)        |
| M2         | Llama-3.1-8B to Qwen2.5-7B (bge-large-en-v1.5)                                | (R,L₀), (R,L₁)        |
| M3         | Qwen2.5-7B to Qwen3-8B (bge-large-en-v1.5)                                    | (R,L₀), (R,L₁)        |
| M4         | bge-large-en v1.0 to v1.5 (Qwen2.5-7B)                                        | (R₀,L), (R₁,L)        |
| M6         | M3 and M4 together                                                            | all four combinations |
| Extensions | M5 (MiniLM to bge-base), NOTES format, mixed-version index, Mistral-7B reader | notebook 04           |

## 7.3 Attacks

- **A1 to A3** (instruction, false fact, false preference): unselected poison. Their flip rates estimate how ordinary stored poison behaves after an upgrade.

- **A4** (upgrade-targeted): phrasings selected to be inert on the old reader and active on a surrogate of the new one. Reported separately, with the number of records for which no phrasing qualified, and evaluated on held-out queries.

- **A5** (scanner-evading): the qualifying phrasing with the lowest Prompt Guard 2 score. It evades the scanner, not the full score.

## 7.4 Methods compared

| **Method**                          | **Construction**                                                                                                          | **Audit**                           |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| Reuse all                           | Nothing quarantined                                                                                                       | Same audit (testing alone)          |
| Reset                               | All memory withheld                                                                                                       | None needed (risk 0, utility floor) |
| Scanner curve                       | Quarantine by Prompt Guard 2 score at every threshold; the 5% benign-quarantine point is highlighted                      | Same audit at each point            |
| Blanket external-source quarantine  | Quarantine every record whose source is external content (perfect labels)                                                 | Same audit                          |
| Authority gating (reference)        | Keep recall; refuse a consequential call whose destination comes only from externally sourced memory (perfect provenance) | Same audit                          |
| Random census                       | B<sub>c</sub> random units probed, positives quarantined, then the score grid                                             | Same audit                          |
| Greedy                              | Top units by w · s probed; no score threshold                                                                             | Same audit                          |
| Scanner score + audit               | Algorithm 1 with the scanner score only                                                                                   | Same audit                          |
| RLMA (full score)                   | Algorithm 1                                                                                                               | Same audit                          |
| RLMA without the differential probe | Ablation                                                                                                                  | Same audit                          |

Every audited method gets the same audit queries and the same total inference budget (score, census and audit together).

## 7.5 Metrics

- Certified result: whether a policy is certified, and its true r<sub>Q</sub>(P<sub>1</sub>, K) on the full test log.

- Utility at the deployed policy: benign task success on the full test log (end to end), split into directory and memory tasks.

- Generalisation: r and benign success on held-out queries.

- Conservative variant: registry-only unauthorised-action rate and its utility cost.

- Cost by component: score construction, attack construction, census, audit, final validation (model calls and generated tokens).

- Migration: transition tables, Equation 5 terms, natural and forced probes reported apart.

## 7.6 Protocol

- **Record once per (policy, query).** Each candidate policy of each method is run once on every test query and every held-out query. Episodes are keyed by what the reader sees, so a query whose context a policy does not change reuses the recorded episode; only queries that retrieved a quarantined record need new runs.

- **Audit replay.** With every candidate's outcomes recorded on the full log, 1,000 audit seeds per cell are replayed on CPU to measure the violation frequency and the certification rate.

- **Grids:** α ∈ {0.01, 0.02, 0.05}; δ = 0.05; G = 5; census sizes B<sub>c</sub> ∈ {0, 50, 100, 200}.

- **Harness:** revision 3 of notebook 01 (AgentDojo's default tool-use instructions, the copy-not-construct rule, one worked example, one call per reply, a nested-call parser fallback, smoke gates, and the system-prompt hash in every episode key).

# 8. Claims and gates

| **Claim**              | **Statement**                                                                                                                                                                                                                                      | **Decided in** | **If it fails**                                                                                                    |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|--------------------------------------------------------------------------------------------------------------------|
| G1 (as pre-registered) | In at least two of M1 to M3, at least 10% of exposed A1 to A3 units inert under the old reader become active under the new one (Wilson lower bound above 5%), with no reader above 5% parse failures. Reported split by natural and forced probes. | 01, 02         | Dig-deeper ladder of plan v1, Section 8, run as development; any adapted setting is confirmed on users 100 to 129. |
| C1 (end to end)        | On the survivor store of at least two of M1 to M3, the new pipeline's attacker-caused rate r<sub>Q</sub> is above 1%, the smallest α.                                                                                                              | 02             | Same ladder; the method is not run on migrations with nothing to remove.                                           |
| C2                     | The scanner curve does not reach r<sub>Q</sub> ≤ 0.01 on any survivor store without quarantining more than 20% of benign exposure.                                                                                                                 | 03             | Report it; restrict the method's claim to the attack families the scanner misses.                                  |
| C3                     | For every migration and α, the violation frequency over audit seeds is not significantly above δ (one-sided exact binomial test at the 1% level).                                                                                                  | 03             | A significant excess means a bug or a broken assumption: found in the code before anything else.                   |
| C4                     | At equal total inference budget, RLMA's certified benign task success is at least that of every other audited method in at least two thirds of migration and α cells.                                                                              | 03             | Reported as found. Ablations show which part, if any, earns its cost.                                              |
| C5                     | Equation 5 terms and transition tables for every migration (descriptive).                                                                                                                                                                          | 02             | Not a gate.                                                                                                        |

No claim requires a baseline to fail. Anything changed after a failed gate counts as development and is confirmed on untouched users before it is reported.

# 9. Notebooks

| **Notebook**            | **Runtime**  | **Content**                                                                                                                                                                                                                                                                                  |
|-------------------------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 00_setup                | CPU          | Done. Repository, suite, harness, Algorithm-1 simulation, tests.                                                                                                                                                                                                                             |
| 01_record_ground_truth  | L4           | Revision 3 running: unit labels for every poisoned unit under the three readers, A4/A5 selection, clean task success, stability, gate G1.                                                                                                                                                    |
| 02_pipelines_and_scores | L4           | Natural/forced split and transition tables from 01's records; held-out queries; end-to-end runs of every pipeline on the unquarantined store and on the survivor stores (C1, C5); registry and canary outcomes; score features including the light probe; score fitted on calibration users. |
| 03_policies_and_audit   | L4, then CPU | Census and candidate policies for every method; every policy recorded end to end on the test and held-out queries; audit replay; C2 to C4; cost by component; scanner curve; authority-gating reference.                                                                                     |
| 04_extensions           | L4           | Ingestion validation; M5; NOTES format; mixed-version index; Mistral-7B reader.                                                                                                                                                                                                              |
| 05_paper_figures        | CPU          | Figures at 600 dpi with the paper's names; tables; numbers file.                                                                                                                                                                                                                             |

Same repository discipline as before: every notebook ends with the standard commit cell; source changes come as %%writefile cells in a stage of the notebook that needs them; runtime version markers rebuild stale records automatically.

# 10. Figures and tables for the paper

- **Figure 1. Framework**, banded: migration; score and census; candidate policies; frozen end-to-end audit; deployed policy with its certificate.

- **Figure 2. Mechanism**: one record that is harmless next to the legitimate payee entry and active once the entry is removed, with both transcripts.

- **Figure 3.** Migration effects: transition tables and Equation 5 terms for each migration.

- **Figure 4.** Benign task success at the certified risk against total inference budget, for every method.

- **Figure 5.** Violation frequency against δ for every cell, and the certification rate.

- **Tables:** migration flips by attack family; methods at one budget; cost by component; held-out generalisation; ablations.

The manuscript carries numbered equations, the threat model and Algorithm 1 from its first draft.

# 11. Risks

| **Risk**                                                                              | **What the plan does about it**                                                                                                                                                           |
|---------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Models still do not consult the directory after revision 3.                           | The smoke gate stops the first session within 40 tasks; the next step is the models' own tool-calling format.                                                                             |
| End-to-end attacker-caused rates on survivor stores are too low to need revalidation. | Claim C1 decides it in notebook 02, before the method's experiments; the dig-deeper ladder runs as development.                                                                           |
| Utility at the certified risk is no better than testing alone.                        | Claim C4 is reported as found; the paper then rests on the migration measurement and the ablations, and is reassessed.                                                                    |
| Recording every policy end to end is expensive.                                       | Episodes are keyed by what the reader sees, so only queries whose context a policy changes are re-run.                                                                                    |
| Run-to-run variation of the serving stack.                                            | Outcomes are defined as recorded greedy executions; the stability study measures variation and is reported.                                                                               |
| Licences.                                                                             | Recorded per checkpoint: Qwen2.5-7B and Qwen3 under Apache 2.0, Qwen2.5-3B under the Qwen research licence, Llama 3.1, Llama 3.2 and Prompt Guard 2 under Meta's licences, BGE under MIT. |

# 12. References

\[1\] Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., Lei, L. Learn then Test: Calibrating predictive algorithms to achieve risk control. arXiv 2110.01052 (v5), Section 2.3.1.

\[2\] Gao, J., Xia, K., Zhang, Y., Hong, X., Lin, M., Wei, X., Li, W., Lu, S. MemPoison: Uncovering persistent memory threats and structural blind spots in LLM agents. arXiv 2607.14651 (v2), 2026.

\[3\] Goyal, A., Ray, J. Does your agent's memory survive a model upgrade? A controlled study of memory portability. arXiv 2609.05339, 2026.

\[4\] Leong, J. W. Injection-execution dissociation: a mechanistic evaluation of persistent memory attacks and defences in stateful LLM agents. arXiv 2605.08442 (v6; the title differs between versions), 2026, Appendix M.1.

\[5\] Louck. Securing LLM-agent long-term memory against poisoning: non-malleable, origin-bound authority with machine-checked guarantees (TMA-NM). arXiv 2606.24322, 2026.

\[6\] Opoku, J., Banahene, D. ToolChain-CRC: Conformal risk control for agentic AI under retrieval and tool-use drift. arXiv 2606.18467, 2026.

\[7\] Ouyang and Hou. MemLineage: Lineage-guided enforcement for LLM agent memory. arXiv 2605.14421, 2026.

\[8\] Pulipaka, S., Hlebik, S., Raghav, L., Abdelnabi, S., Raina, V., Sheth, I., Fritz, M. Hidden in memory: Sleeper memory poisoning in LLM agents. arXiv 2605.15338 (v2), 2026.

\[9\] Sharma. SMSR: Certified defence against runtime memory poisoning in persistent LLM agent systems. arXiv 2606.12703, 2026.

\[10\] Shekhar, S., Xu, Z., Lipton, Z. C., Liang, P. J., Ramdas, A. Risk-limiting financial audits via weighted sampling without replacement. arXiv 2305.06884, 2023, Appendix C.

\[11\] Debenedetti, E. et al. AgentDojo: A dynamic environment to evaluate prompt injection attacks and defenses for LLM agents. NeurIPS 2024 Datasets and Benchmarks; default system message, agentdojo 0.1.35.

\[12\] Yang, A. et al. Qwen2.5 technical report. arXiv 2412.15115. Qwen3 technical report. arXiv 2505.09388.

\[13\] Grattafiori, A. et al. The Llama 3 herd of models. arXiv 2407.21783.

\[14\] Xiao, S. et al. C-Pack: Packaged resources to advance general Chinese embedding (BGE). arXiv 2309.07597.

\[15\] Kwon, W. et al. Efficient memory management for large language model serving with PagedAttention (vLLM). SOSP 2023. arXiv 2309.06180.
