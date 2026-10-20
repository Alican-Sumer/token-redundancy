# Project Plan: Does Attention-Head Pruning Change Token Redundancy in BERT?

## 1. Summary

We fine-tune BERT-base on paraphrase detection (MRPC). Then we ask three things: which tokens become redundant at intermediate layers, whether three cheap signals can predict that redundancy, and whether those signals stay reliable after we prune 25% and 50% of the model's attention heads. The project connects two efficiency techniques that are usually studied separately: **model pruning** (removing heads) and **token pruning** (dropping tokens mid-network).

## 2. Research Questions

1. **RQ1: Signals.** Do attention received, hidden-state change, or hidden-state norm predict which tokens are redundant in the unpruned model?
2. **RQ2: Robustness.** Do those signals stay predictive after attention-head pruning?
3. **RQ3: Dynamics.** Does head pruning change *which* tokens are redundant, or *how many* are redundant at each layer?

**Hypothesis.** Attention received will lose the most predictive power after head pruning, because pruning directly changes the attention patterns it is computed from. The hidden-state signals should be more stable.

## 3. Setup

### Task and data
- **Task:** Paraphrase detection (binary: are the two sentences paraphrases?)
- **Dataset:** MRPC from GLUE (HuggingFace `glue/mrpc`)
  - Train: 3,668 pairs. Hold out 10% as a validation split for hyperparameters and early stopping.
  - Evaluation: the official dev set (408 pairs). GLUE test labels are not public.
- **Input:** `[CLS] sentence1 [SEP] sentence2 [SEP]`, max length 128

### Model
- `bert-base-uncased`: 12 layers, 12 heads per layer, 144 heads total

### Fine-tuning (all models)
- Learning rate 2e-5, batch size 32, 3–5 epochs, AdamW with linear warmup (10%)
- **3 random seeds** per configuration, since MRPC results vary a lot between seeds

## 4. Models Compared

| Model | Heads removed | How |
|---|---|---|
| **M0** (baseline) | 0% (144 heads) | Standard fine-tuning |
| **M25** | 25% (36 heads) | Prune M0, then fine-tune 1–2 more epochs |
| **M50** | 50% (72 heads) | Prune M0, then fine-tune 1–2 more epochs |

**Pruning method:** gradient-based head importance from Michel et al. (2019), *"Are Sixteen Heads Really Better than One?"*
1. Compute each head's importance score on the training split.
2. Remove the lowest-scoring heads globally across all layers, keeping **at least one head per layer**.
3. Physically remove them with HuggingFace `model.prune_heads()`, which gives real FLOP and latency savings.

## 5. Definition of Redundant

A token is **redundant at layer *l*** if deleting it after layer *l* barely changes the model's prediction:

> |p(ŷ | full input) − p(ŷ | token *i* removed after layer *l*)| < ε

- ŷ is the class the model predicts on the full input.
- **ε = 0.01** is the main threshold. We also report ε = 0.005 and ε = 0.02 as a sensitivity check.
- **Drop layers:** *l* ∈ {4, 6, 8}, meaning the token is removed from the output of layer *l* and does not enter layer *l*+1.
- **Candidate tokens:** all tokens except `[CLS]` and `[SEP]`, which are never dropped.

**Implementation:**
1. Run layers 1..*l* normally.
2. Delete token *i* from the hidden states and the attention mask.
3. Run layers *l*+1..12, then the pooler and classifier, by looping over `model.bert.encoder.layer`.

**Cost:** about 408 pairs × ~50 tokens × 3 layers ≈ 61k partial forward passes per model, so roughly 180k for all three models. This is feasible on one GPU with batching. Cache the layer-*l* hidden states and reuse them.

We store the raw |Δp| value for each (token, layer). This gives us both the binary label (redundant or not) and a continuous score for correlation.

## 6. Signals

All signals for a token at layer *l* are computed only from information available at layer *l*, so they could be used for real early token dropping.

| Signal | Formula at layer *l* | Assumed direction |
|---|---|---|
| **Attention received** | Mean over remaining heads of the total attention the token receives from all query positions. Averaging over remaining heads normalizes for head count after pruning. | Low → redundant |
| **Hidden-state change** | 1 − cos(h_{l−1}, h_l) | Low → redundant |
| **Hidden-state norm** | ‖h_l‖₂ | Fixed on the training split (see note) |
| **Random** (baseline) | Uniform random score | — |

**Note on direction:** fix each signal's orientation on the training split before looking at dev results. This avoids choosing whichever direction happens to score better on the evaluation set.

## 7. Metrics

### Task performance and efficiency (for M0, M25, M50)
- **Accuracy and F1:** the standard MRPC metrics
- **FLOPs per example:** computed analytically from head and token counts, or with `fvcore`
- **Inference latency:** ms per batch at batch size 32 and length 128, on a fixed GPU, with warmup, averaged over 100 runs
- **Parameter count**

### Signal quality (RQ1, RQ2)
- **AUROC (primary):** how well each signal separates redundant from non-redundant tokens, using the binary labels from Section 5. 0.5 is random and 1.0 is perfect. Report it per model and per drop layer.
- **Spearman ρ (secondary):** rank correlation between each signal and the raw |Δp|. This catches signals that track importance without a clean threshold.

### Before vs. after pruning (RQ2, RQ3)
- **ΔAUROC:** AUROC on the pruned model minus AUROC on M0, for each signal and layer. This is the main result for RQ2.
- **Redundancy rate:** the fraction of candidate tokens that are redundant at each drop layer. Does pruning make tokens redundant earlier or later?
- **Jaccard overlap:** overlap between the redundant-token sets of M0 and of each pruned model, on the same examples. Does pruning change *which* tokens are redundant?

### Statistical reliability
- Report the mean ± standard deviation over 3 seeds.
- Compute 95% bootstrap confidence intervals over dev examples for AUROC (1,000 resamples).

## 8. Expected Results Tables and Figures

1. **Table 1:** Accuracy, F1, FLOPs, latency, and parameters for M0, M25, and M50
2. **Table 2:** AUROC for each signal × model × layer (with random baseline)
3. **Figure 1:** ΔAUROC per signal, grouped by layer
4. **Figure 2:** Redundancy rate vs. layer, one line per model
5. **Table 3:** Jaccard overlap (M0 vs. M25, M0 vs. M50) per layer
6. **Figure 3 (qualitative):** 2–3 example sentence pairs with redundant tokens highlighted, comparing M0 and M50

## 9. Timeline

| Phase | Work | Output |
|---|---|---|
| **Week 1** | Set up the environment. Fine-tune M0 (3 seeds). Implement mid-layer token removal and verify it: removing no tokens must reproduce the original outputs exactly. | M0 checkpoints, a tested removal function |
| **Week 2** | Compute leave-one-out labels and the three signals for M0. Compute AUROC and Spearman. | RQ1 results |
| **Week 3** | Compute head importance. Create M25 and M50, fine-tune them, and measure accuracy and efficiency. | Pruned checkpoints, Table 1 |
| **Week 4** | Repeat the label and signal computation on M25 and M50. | Raw data for RQ2 and RQ3 |
| **Week 5** | Compute ΔAUROC, redundancy rates, Jaccard, and bootstrap CIs. Make the figures. | Tables 2–3, Figures 1–3 |
| **Week 6** | Write the report and presentation. | Final deliverables |

Compress or stretch the weeks to fit the course deadline.

## 10. Division of Work (two people)

- **Person A:** fine-tuning, head pruning, efficiency measurement (Sections 4 and 7.1)
- **Person B:** token-removal code, leave-one-out labels, signal extraction (Sections 5 and 6)
- **Shared:** analysis, figures, write-up

## 11. Tools

- `torch`, `transformers`, `datasets`
- `scikit-learn` (AUROC), `scipy` (Spearman), `numpy`
- `fvcore` or a hand calculation for FLOPs
- `matplotlib` or `seaborn` for figures

## 12. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| MRPC results are noisy because the dev set is small | 3 seeds, bootstrap CIs, ε sensitivity check |
| Very few tokens fall under ε, or nearly all do | Report the redundancy rate first and check the ε sensitivity range. Change ε only on the training split, never on dev. |
| Pruned models lose too much accuracy | Add 1–2 epochs of recovery fine-tuning. If M50 collapses, use 40% instead. |
| Global pruning empties a layer | Keep at least one head per layer (already in the method) |
| Leave-one-out is too slow | Cache layer-*l* hidden states, batch the removals, or evaluate on a fixed 200-pair subset |
| Attention to `[SEP]` dominates the attention signal | `[SEP]` is excluded from the candidates. Optionally, also exclude attention from `[SEP]` and `[CLS]` queries as a variant. |

## 13. Stretch Goals (only if time allows)

- **Practical token dropping:** at each layer, drop the k% lowest-scoring tokens under the best signal. Plot accuracy vs. keep-ratio and report the combined FLOP savings of head and token pruning.
- Add SST-2 as a single-sentence comparison task.

## 14. Key References

- Devlin et al., 2019. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
- Michel, Levy & Neubig, 2019. *Are Sixteen Heads Really Better than One?*
- Voita et al., 2019. *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.*
- Goyal et al., 2020. *PoWER-BERT: Accelerating BERT Inference via Progressive Word-vector Elimination.*
- Kim et al., 2022. *Learned Token Pruning for Transformers* (LTP).
- Bolya et al., 2023. *Token Merging: Your ViT but Faster* (ToMe).
