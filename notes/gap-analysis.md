# Gap Analysis — Phase 0

**Date:** 2026-07-18 · **Re-run:** quarterly (next: 2026-10-18)
**Roadmap tasks:** #1, #6, #7

## Self-assessment (honest scores)

| Dimension | Level today | Evidence |
|---|---|---|
| Software engineering | ● ● ● ● ● Strong | Years of production experience — the foundation everything builds on |
| Distributed systems | ● ● ● ● ○ Strong | Worked at scale — major leverage for Phases 3 (DDP), 4B (serving), 5 |
| Docker / CI-CD / cloud | ● ● ● ● ○ Strong | Production use |
| Kubernetes | ● ● ○ ○ ○ Gap | Little/none — needed for Phase 5 GPU serving; CKA is *relevant* for me |
| Math (LA/calc/prob/stats) | ● ● ○ ○ ○ Gap | Can follow when explained, can't produce. Phase 1 is NOT skippable for me |
| Classical ML | ● ○ ○ ○ ○ Gap | Tutorials only; no real models, no evaluation discipline yet |
| Deep learning / PyTorch | ○ ○ ○ ○ ○ Gap | Never used PyTorch. Phase 3 in full |
| LLM application layer | ● ● ● ○ ○ Head start | Have called APIs *and built RAG/agent systems* — real working knowledge |
| LLM internals / fine-tuning / serving | ● ○ ○ ○ ○ Gap | Built on top of models, not beneath them: no fine-tuning, self-hosting, KV-cache-level understanding |
| Evaluation (ML + LLM) | ● ○ ○ ○ ○ Gap | Likely built RAG without a rigorous eval harness — the #1 senior differentiator to fix |
| Research literacy | ● ○ ○ ○ ○ Gap | Not reading papers yet |

## My three biggest levers

1. **Distributed systems at scale** — makes Phases 4B/5 (serving, distributed training/inference) far cheaper for me than for most learners. Infra projects (P14, P15) will be my standout portfolio pieces.
2. **Existing RAG/agent experience** — I'm not starting Phase 4C cold. My job there is to *upgrade* intuition into rigor: add hybrid search, reranking, and above all a real eval harness to what I already know how to wire up.
3. **Production shipping habits** — testing, monitoring, deployment discipline transfer directly.

## My three biggest risks

1. **Math is follow-only.** I can't currently derive backprop or reason about LoRA's low-rank math. Phase 1 must be done actively (implement everything), not watched passively.
2. **Zero PyTorch.** The entire Phase 3→4 internals track depends on it. Karpathy Zero-to-Hero is my single most important commitment.
3. **⚠️ Hours: 8–10/week is below the roadmap's 12–18 assumption.** Reality check, not judgment — see below.

## Timeline reality check (the honest part)

The roadmap estimates ~900–1,300 focused hours. At 8–10 hrs/week:
- **Full maximal scope:** ~2+ years. Probably too long — the field moves.
- **Trimmed critical path:** Phase 1 (full, ~5 wks) → Phase 2 (efficient, ~5 wks) → Phase 3 (full, ~10 wks) → Phase 4 (head-start-adjusted, ~12 wks) → Phase 5 (K8s-delta only, ~5 wks) → Phase 6 + 11 (~10 wks) ≈ **~47 weeks of content ≈ 16–20 months at my pace.** Feasible.

**Decisions this forces:**
- **Portfolio: 4 projects + capstone, not 15.** Mine: P4 Production RAG (upgrade what I've built with evals) → P8 Fine-tuning → P14 Distributed inference (plays to my strength) → P15 Observability → P16 capstone (reduced scope).
- **Target role: product/applied AI engineer roles first**, not research labs. My RAG/agent + distsys profile fits them; lab-level internals depth (Phase 4D deep, Phase 7 heavy) costs hours I don't have. Revisit at the quarterly re-run.
- **Certs: skip all** except possibly CKA (it's a *real* gap for me, unlike most cert candidates). Decide at Month 10.
- **OSS: one project only** (LiteLLM or vLLM docs→bugs), starting ~Month 6.
- **Guard the Sunday review religiously** — at 8–10 hrs/wk there's no slack for drift.

## Phase-by-phase adjustment for me

| Phase | Standard plan | My adjustment |
|---|---|---|
| 1 Math | 4–6 wks, compressible | **Full 5–6 wks, active.** Not compressible for me |
| 2 Classical ML | 5–7 wks | ~5 wks; focus evaluation/leakage discipline over breadth |
| 3 Deep learning | 8–10 wks | **Full 10 wks — my critical gap.** Karpathy end-to-end |
| 4 LLMs | 12–16 wks | ~12 wks: light on 4C basics (know it), **heavy on internals, serving, fine-tuning, and EVALS** |
| 5 Infra | 6–8 wks | ~5 wks: mostly the **K8s+GPU delta**; distsys concepts already known |
| 6 Design + 11 Interview | ~15 wks combined | Full — this is where offers are won |
| 7 Papers | ongoing | 1 paper/2 weeks (not 1/week) — pace-adjusted |
| 10 Certs | optional | Skip, except maybe CKA (Month 10 decision) |

## First 4 weeks (starting Monday)

- **Wk 1:** GPU + env setup (tasks #2–#5, #8) · 3B1B Linear Algebra (all) · implement matmul + PCA-via-SVD in NumPy (#12–#13)
- **Wk 2:** 3B1B Calculus · MML Part I selected · SVD image compression + LoRA-style decomposition by hand (#15–#16)
- **Wk 3:** **Backprop from scratch + gradient check (#17–#18)** — the keystone exercise
- **Wk 4:** Stat 110 selected + StatQuest core · MLE, naive Bayes, bias–variance sim (#21–#27)

Every implementation gets an experiment note (`./scripts/new.sh experiment "..."`). First weekly review: this Sunday.

## What "closed" looks like (re-check quarterly)

- [ ] Math: can derive backprop and explain LoRA's math unaided
- [ ] PyTorch: can write a training loop with mixed precision from memory
- [ ] Evaluation: my RAG work has a measured eval harness with numbers
- [ ] Internals: can explain KV cache sizing and cut inference cost on a whiteboard
- [ ] K8s: can deploy a GPU-served model with autoscaling
