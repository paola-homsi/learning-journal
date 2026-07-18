# The Senior AI Engineer Roadmap
### For the Experienced Software Engineer → Elite Production AI Engineer (12–18 Months)

> **Who this is for:** You already ship production software. You know systems, APIs, databases, cloud, containers, CI/CD, distributed systems, and Python. This roadmap does **not** teach you to code. It converts a strong backend/systems engineer into a Senior AI Engineer who can design, build, deploy, scale, and reason about production AI — and pass Senior AI Engineer interviews at OpenAI, Anthropic, DeepMind, Meta, Nvidia, Databricks, Snowflake, Scale AI, Palantir, and peers.
>
> **The core thesis:** The market rewards **AI *engineers*, not AI *researchers*.** ~80% of the value you'll deliver (and be interviewed on) is *systems* work — retrieval, serving, evaluation, cost/latency, reliability — wrapped around models you mostly *don't* train from scratch. This roadmap is weighted accordingly: deep enough theory to reason about architectures, but relentlessly biased toward **shipping**.
>
> **How to use it:** Phases are *mostly* sequential but overlap heavily (see the dependency graph in Final Deliverables). If you're time-constrained, the **critical path** is: Phase 1 (light) → Phase 3 → Phase 4 → Phase 5/6 → Phase 11. Phases 2, 7, 9, 10 are parallelizable or optional-depending-on-target-role.

---

## Table of Contents

- [Phase 0 — Gap Analysis](#phase-0--gap-analysis)
- [Phase 1 — Mathematical Foundations](#phase-1--mathematical-foundations)
- [Phase 2 — Machine Learning Foundations](#phase-2--machine-learning-foundations)
- [Phase 3 — Deep Learning](#phase-3--deep-learning)
- [Phase 4 — Large Language Models](#phase-4--large-language-models)
- [Phase 5 — AI Infrastructure](#phase-5--ai-infrastructure)
- [Phase 6 — AI System Design](#phase-6--ai-system-design)
- [Phase 7 — AI Research Literacy](#phase-7--ai-research-literacy)
- [Phase 8 — Hands-on Portfolio](#phase-8--hands-on-portfolio)
- [Phase 9 — Open Source Contributions](#phase-9--open-source-contributions)
- [Phase 10 — Certifications](#phase-10--certifications)
- [Phase 11 — Interview Preparation](#phase-11--senior-ai-engineer-interview-preparation)
- [Phase 12 — Weekly Learning Schedule](#phase-12--weekly-learning-schedule)
- [Final Deliverables](#final-deliverables)
  - [1. Dependency Graph](#1-dependency-graph)
  - [2. Roadmap Table](#2-roadmap-table)
  - [3. 300+ Task Checklist](#3-300-task-checklist)
  - [4. Portfolio Roadmap](#4-portfolio-roadmap)
  - [5. Reading Roadmap](#5-reading-roadmap)
  - [6. Interview Roadmap](#6-interview-roadmap)
  - [7. Certification Roadmap](#7-certification-roadmap)
  - [8. Top 100 Skills for 2026](#8-top-100-ai-engineering-skills-2026)
  - [9. Why These Choices](#9-why-these-choices-over-alternatives)

---

# Phase 0 — Gap Analysis

**Duration:** 2–3 days (assessment + baseline setup). **Do this honestly — it determines how much of Phase 1–2 you can skip.**

### What you already have (and can leverage hard)

| Skill | How it transfers to AI engineering |
|---|---|
| Production software delivery | **Your #1 advantage.** Most ML people can't ship. You already can. AI systems are 90% software engineering. |
| System design | Directly reusable. AI system design = system design + retrieval/serving/eval primitives. |
| APIs, databases, cloud, containers, CI/CD | Reused wholesale. Vector DBs are still DBs; model serving is still a service; GPU nodes are still nodes. |
| Distributed systems | Enormous edge for distributed training/inference, sharding, and data pipelines. |
| Python fluency | Everything runs on it. You skip the language-learning tax entirely. |
| Monitoring/observability | Transfers to model monitoring, drift detection, eval pipelines. |
| Testing discipline | Rare and prized in ML. Applies to eval harnesses and regression suites. |

### What you likely lack (the actual gap)

| Gap | Severity | Where addressed |
|---|---|---|
| Math *fluency* for reasoning about models (not just knowing terms) | Medium | Phase 1 |
| Statistical thinking (bias/variance, leakage, proper evaluation) | **High** — most common failure in ML interviews | Phase 2 |
| Deep learning mechanics (backprop, autograd, training loops) | **High** | Phase 3 |
| PyTorch as a *native* tool | **High** | Phase 3 |
| Transformer internals (attention, KV cache, RoPE) | **Critical** | Phase 4 |
| LLM systems (RAG, agents, serving, quantization, fine-tuning) | **Critical — this is the job** | Phase 4, 5, 6 |
| GPU/CUDA mental model (memory, throughput, batching) | Medium–High | Phase 3, 5 |
| Research literacy (reading papers, tracking SOTA) | Medium | Phase 7 |
| ML-specific interview patterns | **High** | Phase 11 |

### Estimated gap

For a strong SWE: **~900–1,300 focused hours** to reach Senior AI Engineer bar. At **12–15 hrs/week that's ~15–18 months**; at **20 hrs/week, ~12 months**. The gap is *narrower than you fear on engineering, wider than you expect on evaluation/theory intuition.*

### Baseline setup (do now)

- [ ] GPU access: a **Colab Pro+** ($50/mo) or **Lambda/RunPod/Vast.ai** on-demand account (an RTX 4090 or A100 for spot work). You cannot learn this on CPU alone.
- [ ] `uv` for Python env management (faster than conda/pip; the field has largely moved to it).
- [ ] Accounts: Hugging Face, Weights & Biases (free tier), GitHub (dedicated learning org/repo).
- [ ] A **learning journal repo** — every project, every paper note, every experiment. This *becomes your portfolio and interview talking points.*

---

# Phase 1 — Mathematical Foundations

**Duration:** 4–6 weeks (compressible to 2–3 if strong; treat as *reference you return to*, not a gate). **Objective:** Not to become a mathematician — to reason fluently about *why* models behave as they do, read papers without stalling, and debug training. **Learn the math you'll actually use; skip the rest.**

> **Opinionated stance:** Do **not** spend 6 months on math you'll never use. An AI *engineer* needs *fluency and intuition*, not proof-writing. Learn it *actively* — implement every concept in NumPy. The single highest-ROI resource is **3Blue1Brown for intuition + implementation for retention.**

### Required knowledge
NumPy fluency, comfort with mathematical notation. That's it.

### 1.1 Linear Algebra — *the language of deep learning*

**Topics:** vectors, matrices, matrix multiplication as the core primitive, eigendecomposition, **SVD** (the most important single tool — PCA, low-rank approximation, LoRA all derive from it), tensor operations & broadcasting, matrix calculus shapes.

- **Intuition:** 3Blue1Brown, *Essence of Linear Algebra* (YouTube, free) — non-negotiable, watch it all.
- **Depth:** MIT 18.06 Gilbert Strang (OCW, free) — *the* linear algebra course. Watch selectively; Strang's *Introduction to Linear Algebra* book if you want rigor.
- **AI-focused:** *Mathematics for Machine Learning* (Deisenroth, Faisal, Ong — **free PDF**, mml-book.github.io) — Part I. Written *for* ML, skips irrelevant material.
- **Exercises:** Implement in pure NumPy: matmul, PCA from scratch via SVD, eigendecomposition of a covariance matrix, low-rank image compression with SVD. *Reproduce a LoRA decomposition by hand.*

### 1.2 Calculus — *how models learn*

**Topics:** derivatives, partial derivatives, **gradients**, the **chain rule** (backprop is literally this), **Jacobians** and their shapes, Hessians (conceptually), optimization landscapes.

- **Intuition:** 3Blue1Brown, *Essence of Calculus* (free).
- **AI-focused:** *Mathematics for ML*, calculus chapters. Focus on **vector-valued** chain rule.
- **The key exercise:** Implement **backpropagation from scratch** for a 2-layer MLP in NumPy — forward pass, manual gradient derivation, gradient check via finite differences. This one exercise teaches more than 20 hours of video.

### 1.3 Probability — *reasoning under uncertainty*

**Topics:** sample spaces, conditional probability, **Bayes' theorem**, common distributions (Bernoulli, Categorical, Gaussian, Poisson), **expectation & variance**, **likelihood & log-likelihood**, MLE/MAP, sampling.

- **Resource:** *Mathematics for ML* Part; Harvard **Stat 110** (Joe Blitzstein, free on YouTube + edX) — best probability course available, worth it if probability is weak.
- **Exercises:** Implement MLE for a Gaussian; sample from distributions and verify moments empirically; build a naive Bayes classifier from scratch.

### 1.4 Statistics — *the source of most ML interview failures*

**Topics:** hypothesis testing, p-values (and their misuse), **confidence intervals**, **bias–variance decomposition** (know this cold — it's asked constantly), A/B testing, bootstrapping, multiple-comparison problems.

- **Resource:** StatQuest with Josh Starmer (YouTube, free) — *the* channel for ML statistics intuition. *Bias–variance, ROC/AUC, cross-validation videos are mandatory.*
- **Exercises:** Bootstrap a confidence interval; run a t-test manually; simulate the bias–variance tradeoff by fitting polynomials of increasing degree.

### 1.5 Optimization — *the engine*

**Topics:** convexity (briefly), **gradient descent**, **SGD**, momentum, **Adam / AdamW** (know *why* AdamW ≠ Adam), learning-rate schedules (warmup, cosine decay), **regularization** (L1/L2, dropout as regularization, weight decay), gradient clipping.

- **Resource:** Sebastian Ruder's *"An overview of gradient descent optimization algorithms"* (blog, free — still the best single reference). *d2l.ai* optimization chapter.
- **Exercises:** Implement SGD, Momentum, and Adam from scratch; visualize their trajectories on a 2D loss surface (e.g., Rosenbrock/Beale). Reproduce a learning-rate warmup+cosine schedule.

### 1.6 Information Theory — *the theory behind LLM training*

**Topics:** **entropy**, **cross-entropy** (your loss function for every classifier and LLM), **KL divergence** (RLHF, DPO, distillation, VAEs all use it), mutual information, perplexity (= exp of cross-entropy; the core LLM metric).

- **Resource:** 3Blue1Brown has touches; Chris Olah's blog *"Visual Information Theory"* (free) — beautiful and sufficient.
- **Exercises:** Compute entropy/cross-entropy/KL between distributions in NumPy; derive why cross-entropy loss + softmax gives clean gradients; compute perplexity of a toy language model.

### Phase 1 Deliverables
- [ ] **`ml-math-from-scratch`** repo: NumPy implementations of PCA/SVD, backprop, Adam, naive Bayes, entropy/KL, bias–variance simulation.
- **Portfolio project:** A Jupyter notebook series "The Math of Deep Learning, Implemented" — publishable as a blog series (great early credibility signal).
- **Interview prep goal:** Explain bias–variance, why we use cross-entropy, what SVD does, and what Adam fixes over SGD — *verbally, in 2 minutes each.*

### Phase 1 Curated Resources
| Type | Pick | Cost | Why |
|---|---|---|---|
| Best free course | 3Blue1Brown (LA + Calc) + StatQuest | Free | Unmatched intuition-building |
| Best book | *Mathematics for Machine Learning* | Free PDF | Written for ML, skips the irrelevant |
| Best rigor | MIT 18.06 (Strang) | Free | The canonical LA course |
| Best probability | Harvard Stat 110 | Free | Best-taught probability anywhere |
| Best blog | Chris Olah + Sebastian Ruder | Free | Reference-grade, still current |
| Paid (only if needed) | *Imperial College Math for ML* (Coursera) | ~$49/mo | Only if you need structure/deadlines; otherwise skip |

---

# Phase 2 — Machine Learning Foundations

**Duration:** 5–7 weeks. **Objective:** Master classical ML — not because it's trendy, but because (a) it's *still* the right tool for most tabular/business problems, (b) it's *heavily interviewed* at every company on your list, and (c) the discipline of **proper evaluation and avoiding data leakage** is the single most valuable transferable skill in all of ML. **Do not skip this to rush to LLMs.**

> **Opinionated stance:** Many self-taught AI engineers have a gaping hole here and it *shows* in interviews. A Senior AI Engineer who can't reason about class imbalance, leakage, or when a gradient-boosted tree beats a neural net is not senior. Gradient boosting (XGBoost/LightGBM) still wins most Kaggle tabular competitions and most real business problems.

### Required knowledge
Phase 1 (probability, statistics, optimization). Python + NumPy/Pandas.

### 2.1 Core paradigms & the ML mindset
- **Supervised learning:** regression, classification; linear/logistic regression *derived* (not just called from sklearn).
- **Unsupervised learning:** clustering (k-means, DBSCAN, hierarchical), dimensionality reduction (PCA, t-SNE, UMAP), anomaly detection (isolation forests, one-class SVM).
- **The bias–variance tradeoff in practice**, underfitting vs overfitting, the **fundamental role of the train/val/test split.**

### 2.2 Tree models & boosting — *the workhorses*
- Decision trees → random forests → **gradient boosting**.
- **XGBoost, LightGBM, CatBoost** — know the differences: LightGBM (leaf-wise, fast, large data), XGBoost (level-wise, robust default), CatBoost (categorical features, less tuning). *When to reach for each.*
- SVMs (kernel trick intuition — still asked).

### 2.3 The evaluation discipline (*the part that makes you senior*)
- **Metrics:** accuracy's failures, precision/recall/F1, **ROC-AUC vs PR-AUC** (and when each lies), log loss, calibration, regression metrics (RMSE/MAE/R²), ranking metrics (NDCG, MAP, MRR).
- **Cross-validation:** k-fold, stratified, **time-series splits** (get this wrong and you leak), group k-fold.
- **Data leakage** — the #1 real-world ML bug. Target leakage, train/test contamination, temporal leakage. *Interviewers love this.*
- **Feature engineering & importance:** encoding, scaling, interactions; permutation importance, SHAP values.
- **Experiment tracking:** Weights & Biases or MLflow from day one.

### 2.4 Recommendation systems (frequently interviewed)
- Collaborative filtering (matrix factorization), content-based, hybrid; the cold-start problem; implicit vs explicit feedback; candidate generation → ranking two-stage architecture (this shows up in system design too).

### 2.5 Production concerns
- Class imbalance (resampling, class weights, threshold tuning), concept drift, model monitoring, retraining triggers, offline vs online metrics divergence.

### Phase 2 Deliverables
- [ ] End-to-end tabular ML project with proper CV, leakage checks, SHAP explanations, and W&B tracking.
- [ ] A recommendation system on a real dataset (MovieLens).
- **Portfolio project:** **"Production-Grade Tabular ML Pipeline"** — a Kaggle competition entry (aim top 25%) *with a writeup* on your evaluation methodology and leakage prevention. The writeup matters more than the rank.
- **Interview prep goal:** Given a business problem, choose the right model + metric + validation strategy and *justify it*; spot leakage in a described pipeline.

### Phase 2 Curated Resources
| Type | Pick | Cost | Why |
|---|---|---|---|
| Best free course | Andrew Ng, *Machine Learning Specialization* (Coursera, audit free) | Free | The canonical intro, modernized (Python, not Octave) |
| Best deeper course | *StatQuest* + Sebastian Raschka's videos/book | Free/Paid | Raschka's *Machine Learning with PyTorch and Scikit-Learn* is the best practitioner book |
| Best book | *Hands-On Machine Learning* (Géron, 3rd ed.) | ~$60 | **Worth every cent** — the single best applied ML book; sklearn + Keras + real projects |
| Best reference | *Introduction to Statistical Learning* (ISLP, free PDF) | Free | Rigorous, Python edition, beautifully written |
| Best docs | scikit-learn user guide | Free | Genuinely excellent, reads like a textbook |
| Best practice | Kaggle Learn + real competitions | Free | The only way to internalize evaluation discipline |
| YouTube | StatQuest (Starmer) | Free | Best ML intuition on the internet |

> **Why *Hands-On ML* over *Pattern Recognition and ML* (Bishop):** Bishop is a beautiful *researcher's* book. You are an *engineer*. Géron teaches you to ship; Bishop teaches you to derive. Read Géron now; keep Bishop as an optional deep-reference.

---

# Phase 3 — Deep Learning

**Duration:** 8–10 weeks. **Objective:** Own the mechanics of neural networks end-to-end — build them from scratch, then master PyTorch as a *native tool*, then understand the training-at-scale concerns that separate engineers from tutorial-followers. **This is where you stop being an ML user and start being an ML engineer.**

> **Opinionated stance:** There is exactly one correct way to *start* this phase: **Andrej Karpathy's "Neural Networks: Zero to Hero."** Build micrograd (an autograd engine) and makemore/nanoGPT with him, by hand. Nothing else on the internet builds the same depth of understanding. Then, and only then, use PyTorch's abstractions — because now you know what they're hiding. **PyTorch, not TensorFlow.** The research world, most of industry, and nearly every company on your target list is PyTorch-first. Learn TF/Keras only if a specific employer requires it.

### Required knowledge
Phases 1–2. Especially: chain rule, gradients, cross-entropy, SGD/Adam.

### 3.1 Neural network fundamentals (build from scratch first)
- Perceptron → MLP; forward pass; **backpropagation via autograd** (build micrograd with Karpathy); activation functions (ReLU, GELU, SiLU — and why), weight initialization (Xavier/He and *why it matters*).
- **Loss functions:** MSE, cross-entropy, and how loss shapes learning.
- **Optimizers in practice:** SGD+momentum vs Adam vs AdamW; when each wins.

### 3.2 Training that actually converges
- **Regularization:** dropout, weight decay, early stopping, data augmentation, label smoothing.
- **Normalization:** BatchNorm, **LayerNorm** (dominant in transformers), RMSNorm; *why* normalization stabilizes training.
- **Residual connections** (the innovation that made deep networks trainable — understand the gradient-flow argument cold).
- The training loop: overfitting a single batch (the #1 debugging technique), learning-rate finding, gradient clipping, checkpointing.

### 3.3 Architectures
- **CNNs:** convolution, pooling, receptive fields; ResNet; when CNNs still matter (vision, though ViTs now compete).
- **RNNs/LSTMs/GRUs:** sequence modeling, vanishing gradients, why they were *replaced* — you need to understand them to understand *why transformers won*, but don't over-invest.
- **Attention → Transformers:** self-attention from scratch, multi-head attention, positional encoding. **This is the bridge to Phase 4** — build it by hand here.
- **Embeddings & tokenization:** what embeddings are geometrically; word2vec intuition; subword tokenization preview.
- **Transfer learning & fine-tuning:** feature extraction vs full fine-tuning; the pretraining→finetuning paradigm.

### 3.4 PyTorch mastery
- Tensors, autograd, `nn.Module`, `DataLoader`, custom datasets, training/eval loops, `torch.save`/checkpointing.
- **Mixed precision** (`torch.cuda.amp` / bf16) — essential for modern training.
- Debugging: hooks, `torchviz`, common shape/device bugs.

### 3.5 Training at scale & GPU literacy
- **Distributed training:** DataParallel vs **DistributedDataParallel (DDP)**; the concepts behind FSDP/ZeRO (data/tensor/pipeline parallelism) — you'll go deeper in Phase 5.
- **GPU mental model:** memory hierarchy, why batching matters, throughput vs latency, what actually consumes VRAM (params + activations + optimizer states + gradients — know this formula).
- **CUDA basics:** enough to reason about kernels and profiling; you don't need to *write* CUDA yet, but understand what a kernel is and why fused kernels (FlashAttention) matter.
- **Profiling:** PyTorch profiler, `nvidia-smi`, identifying bottlenecks (data loading vs compute vs communication).

### Phase 3 Deliverables
- [ ] **`micrograd`** and **`makemore`/nanoGPT`** built alongside Karpathy (in your repo, with your notes).
- [ ] An image classifier trained with proper augmentation, mixed precision, and W&B tracking.
- [ ] A transformer built from scratch (char-level), trained on a small corpus.
- [ ] A DDP training script that runs on 2+ GPUs.
- **Portfolio project:** **"Transformer From Scratch"** — a clean, documented, from-scratch implementation of a GPT-style model with a blog writeup explaining every component. *This is one of the strongest possible portfolio pieces for an LLM-focused role.*
- **Interview prep goal:** Whiteboard backprop; explain attention math; debug a described training failure (loss NaN, not converging, overfitting); explain what consumes GPU memory during training.

### Phase 3 Curated Resources
| Type | Pick | Cost | Why |
|---|---|---|---|
| **Best resource, period** | Karpathy, *Neural Networks: Zero to Hero* (YouTube) | Free | Unmatched depth-from-first-principles; build everything by hand |
| Best structured course | **fast.ai** *Practical Deep Learning* (Jeremy Howard) | Free | Top-down, ships fast, brilliant pedagogy |
| Best theory course | *NYU Deep Learning* (LeCun & Canziani) or CMU 11-785 | Free | Rigorous; LeCun's own course |
| Best interactive book | **Dive into Deep Learning (d2l.ai)** | Free | Runnable, PyTorch+, comprehensive, updated |
| Best book | Raschka, *Machine Learning with PyTorch and Scikit-Learn* | ~$50 | Best PyTorch practitioner book |
| Best DL theory book | *Deep Learning* (Goodfellow et al., free online) | Free | The reference; dense, use selectively |
| Best docs | PyTorch tutorials + docs | Free | Excellent, hands-on |
| Best playlist | Karpathy + StatQuest DL series | Free | — |

> **Why Karpathy over the Coursera *Deep Learning Specialization*:** Ng's DL Spec is good and structured, but it abstracts away the guts. Karpathy makes you *build the autograd engine*. For an engineer who wants to *reason about new architectures* (your stated goal), the from-scratch path is strictly superior. Do the DL Spec only if you need external structure/certificate.


---

# Phase 4 — Large Language Models

**Duration:** 12–16 weeks (the deepest, most important phase — this *is* the job for most Senior AI Engineer roles in 2026). **Objective:** Go from understanding transformers to being able to architect, serve, evaluate, optimize, and fine-tune LLM systems in production, and to reason about frontier architectures well enough to discuss them in interviews at the labs building them.

> **Opinionated stance:** This is where you spend the most time and build the most portfolio. The field splits into two competencies: **(A) LLM *systems* engineering** (RAG, agents, serving, evaluation, cost/latency) — which is ~80% of jobs and interviews — and **(B) LLM *internals & training*** (architecture, fine-tuning, alignment) — which is the differentiator for lab roles. Do both, but weight toward (A) unless targeting a research-adjacent role. **The single most underrated skill: evaluation.** Anyone can wire up a RAG demo; a senior engineer can *tell you if it's actually working* and *make it cheaper and faster.***

### Required knowledge
Phase 3 (transformers from scratch, PyTorch, attention). This phase assumes you can read `nn.Module` code fluently.

---

## Part A — Transformer Internals (go deep)

### 4A.1 The architecture, thoroughly
- **Self-attention** revisited at production depth: Q/K/V, scaled dot-product, causal masking, multi-head, **multi-query (MQA)** and **grouped-query attention (GQA)** — and *why* they exist (KV-cache memory).
- **Positional encodings:** absolute (sinusoidal), learned, **RoPE (Rotary)** — dominant today, understand the rotation intuition — and ALiBi. *Why position matters and how RoPE enables length extrapolation.*
- **The KV cache:** what it is, why it dominates inference memory, how its size scales (layers × heads × seq_len × dim × 2 × dtype). *This is asked in every serious LLM systems interview.*
- **FlashAttention** (v1→v3): the IO-aware, fused-kernel insight; why it's faster *and* more memory-efficient; you don't implement it but you must explain it.
- **Normalization & activations in LLMs:** pre-norm vs post-norm, RMSNorm, SwiGLU/GeGLU.
- **MoE (Mixture of Experts):** sparse activation, routing, why it decouples params from FLOPs (Mixtral, DeepSeek-V3, and most frontier models use it).

### 4A.2 Tokenization
- **BPE**, WordPiece, **SentencePiece**, tiktoken; byte-level BPE; vocabulary tradeoffs; why tokenization causes weird failures (arithmetic, spelling, non-English). Build a BPE tokenizer from scratch (Karpathy has a video).

### 4A.3 Training dynamics (conceptual for engineers, deep for lab roles)
- Pretraining objectives (next-token prediction), scaling laws (Chinchilla — compute-optimal training), the pretraining→SFT→RLHF/DPO pipeline, data curation's outsized role.

---

## Part B — LLM Inference, Serving & Optimization (*core engineering*)

### 4B.1 Inference & serving stacks
- **vLLM** (PagedAttention, continuous batching — learn this well; it's the industry default), **SGLang** (RadixAttention, structured gen), **TensorRT-LLM** (Nvidia, max perf), **TGI**, **llama.cpp/Ollama** (local/edge).
- **Continuous/dynamic batching**, **PagedAttention**, prefill vs decode phases, speculative decoding, prefix caching.
- Throughput vs latency tradeoffs; TTFT (time-to-first-token) vs TPOT (time-per-output-token); how to actually load-test an LLM endpoint.

### 4B.2 Quantization & compression
- **Quantization:** INT8/INT4, **GPTQ, AWQ, GGUF** (llama.cpp format), bitsandbytes, FP8; weight-only vs weight+activation; the quality/speed/memory tradeoff.
- **Formats:** GGUF, **ONNX** (interop/runtime), safetensors.
- Distillation and pruning (conceptually).

### 4B.3 Cost & latency optimization (*a senior differentiator*)
- Model selection by task (don't use GPT-4-class for everything), routing/cascades (cheap model first, escalate), prompt caching, semantic caching, batching strategies, KV-cache reuse, right-sizing context, streaming.
- **Build a cost model** for an LLM feature (tokens × price, cache hit rate, model mix). Interviewers respect this.

---

## Part C — LLM Application Engineering (*the bulk of jobs*)

### 4C.1 Prompt & context engineering
- Prompting techniques (few-shot, chain-of-thought, ReAct, self-consistency) — and their *limits*.
- **Structured outputs:** JSON mode, constrained decoding, grammar-based generation, Pydantic/Instructor, **function/tool calling.**
- **Context engineering:** the discipline of assembling the right context window — retrieval + memory + instructions + tools — under a token budget. (The field has largely reframed "prompt engineering" as "context engineering.")

### 4C.2 RAG (Retrieval-Augmented Generation) — *the most-built LLM system*
- **The full pipeline:** ingestion → **chunking** (fixed, semantic, recursive, late-chunking) → **embedding** → indexing → retrieval → reranking → generation.
- **Embedding models:** how to choose (MTEB leaderboard), open (BGE, E5, Nomic, GTE) vs API (OpenAI, Cohere, Voyage), dimension/latency/quality tradeoffs, fine-tuning embeddings.
- **Vector databases:** pgvector, Qdrant, Weaviate, Milvus, Pinecone, LanceDB, FAISS — *and when you don't need one* (small corpora, in-memory).
- **Hybrid search:** dense + sparse (BM25), reciprocal rank fusion; **reranking** (cross-encoders, Cohere Rerank) — usually the highest-ROI RAG improvement.
- **Advanced RAG:** query rewriting, HyDE, multi-query, parent-document retrieval, contextual retrieval (Anthropic's approach), **GraphRAG** (knowledge-graph-augmented).
- **RAG evaluation:** RAGAS, faithfulness, context relevance, answer relevance; retrieval metrics (recall@k, MRR, NDCG); *how to know your RAG is actually good.*

### 4C.3 Agents & tool use (*the 2025–2026 frontier*)
- **Agentic patterns:** ReAct, plan-and-execute, reflection, tool use, function calling loops.
- **Multi-agent systems:** orchestrator-worker, debate, hierarchical; when multi-agent helps vs adds fragility (be skeptical — many multi-agent designs are worse than a good single agent).
- **Model Context Protocol (MCP):** the emerging standard for connecting models to tools/data; understand the client-server architecture, why it matters for interoperability, and how to build an MCP server.
- **Memory:** short-term (context), long-term (vector/DB-backed), episodic vs semantic memory, memory management strategies.
- **Long context:** when to use long-context models vs RAG; the "lost in the middle" problem; context rot.
- **Agent evaluation & reliability:** the hard part — tool-call accuracy, task completion, cost/step, guardrails, human-in-the-loop.
- **Frameworks:** LangGraph, LlamaIndex, CrewAI, AutoGen, PydanticAI, OpenAI Agents SDK, Smol Agents — *learn one deeply, understand the tradeoffs; know when to use no framework at all* (many seniors roll their own for control).

### 4C.4 Guardrails, safety & reliability
- Input/output validation, PII detection/redaction, prompt-injection defense (critical for agents with tools), jailbreak resistance, content moderation, **hallucination mitigation** (grounding, citations, verification, abstention), NeMo Guardrails / Guardrails AI.

### 4C.5 Evaluation (*the skill that makes you senior — cannot be overstated*)
- **LLM-as-judge** (and its biases — position bias, verbosity bias, self-preference), pairwise comparison, rubric grading.
- Offline eval sets, **golden datasets**, regression testing for prompts/models, A/B testing in production.
- Benchmarks and their limits: MMLU, GPQA, HumanEval/MBPP, SWE-bench, MT-Bench, Chatbot Arena, and *why leaderboard rank ≠ your-task performance.*
- Tools: LangSmith, LangFuse, Braintrust, Phoenix/Arize, OpenAI Evals, Inspect (UK AISI).
- **Build an eval harness** for one of your projects — this alone will impress interviewers more than any demo.

---

## Part D — Fine-tuning & Alignment (differentiator for lab/advanced roles)

### 4D.1 When to fine-tune (and when *not* to)
- The decision tree: prompting → RAG → fine-tuning. Fine-tuning is for *behavior/format/domain-style*, rarely for *knowledge* (that's RAG's job). Knowing this distinction is a senior signal.

### 4D.2 Parameter-efficient fine-tuning
- **LoRA** (low-rank adaptation — understand the math, it's your SVD/low-rank knowledge applied), **QLoRA** (4-bit + LoRA, democratized fine-tuning), **PEFT** library, adapters, prefix/prompt tuning.
- **Full fine-tuning** vs PEFT tradeoffs; catastrophic forgetting.
- Practical stacks: **Axolotl**, **Unsloth** (fast/memory-efficient), Hugging Face **TRL**, Llama-Factory.

### 4D.3 Preference tuning & alignment
- **RLHF** (reward modeling + PPO) — the InstructGPT pipeline; understand it conceptually even if you never run it.
- **DPO** (Direct Preference Optimization — simpler, no reward model, now dominant), and successors (IPO, KTO, ORPO, GRPO — the DeepSeek reasoning approach).
- Constitutional AI (Anthropic's approach), RLAIF.
- Alignment concepts: helpfulness/harmlessness/honesty, reward hacking, why alignment is hard.

### 4D.4 Reasoning models (2025–2026 hot topic)
- Test-time compute / inference-time scaling, chain-of-thought RL (o1/o3, DeepSeek-R1, GRPO), when reasoning models are worth their latency/cost.

### Phase 4 Deliverables
- [ ] BPE tokenizer from scratch.
- [ ] A production RAG system with hybrid search, reranking, and a RAGAS eval harness.
- [ ] An agent with tool use + an MCP server you built.
- [ ] A LoRA/QLoRA fine-tune of an open model (e.g., Llama/Qwen/Mistral) on a domain task, with before/after eval.
- [ ] An LLM served via vLLM with a documented cost/latency benchmark and a quantized variant.
- [ ] An LLM-as-judge eval pipeline.
- **Portfolio projects (pick 3–4 to polish):** Production RAG, Multi-agent workflow, Fine-tuning project, Serving/quantization benchmark (all detailed in Phase 8).
- **Interview prep goal:** Design a RAG system end-to-end on a whiteboard; explain the KV cache and how you'd cut inference cost 5×; decide fine-tune vs RAG vs prompt for a given problem *and defend it*; explain LoRA's math; critique a multi-agent design; describe how you'd evaluate an LLM feature you can't fully specify.

### Phase 4 Curated Resources
| Type | Pick | Cost | Why |
|---|---|---|---|
| **Best free deep-dive** | Karpathy: *Let's build GPT*, *Let's build the GPT Tokenizer*, *Let's reproduce GPT-2*, **"Deep Dive into LLMs like ChatGPT"** (2025) | Free | The definitive from-scratch LLM education |
| **Best conceptual blog** | **Lilian Weng's blog** (lilianweng.github.io) | Free | *The* reference for agents, prompting, RLHF, hallucination — read all of it |
| Best visual explainer | Jay Alammar, *The Illustrated Transformer* + his book *Hands-On Large Language Models* | Free / ~$50 | Best visual intuition; the book is excellent and current |
| Best practical book | **Sebastian Raschka, *Build a Large Language Model (From Scratch)*** | ~$50 | Builds a GPT + fine-tuning from zero; superb for engineers |
| Best free course | **Hugging Face LLM Course** + **Agents Course** | Free | Hands-on, current, industry-standard tooling |
| Best RAG/eval resource | LlamaIndex docs + RAGAS docs + LangChain docs | Free | The practitioner references |
| Best fine-tuning | HF PEFT/TRL docs, Unsloth notebooks, Axolotl examples | Free | Runnable, current |
| Best serving | vLLM docs + blog | Free | The serving standard |
| Best applied course (paid) | **DeepLearning.AI short courses** (RAG, Agents, Fine-tuning, Functions, Evals) | Free (currently) | Short, high-signal, built with the tool-makers |
| Best "state of the field" | Simon Willison's blog, Chip Huyen's blog + *AI Engineering* (O'Reilly 2025 book) | Free / ~$50 | **Chip Huyen's *AI Engineering* is arguably the single best book for this exact role** — read it |
| Best community/news | Latent Space podcast/newsletter, The Batch (DeepLearning.AI) | Free | Stay current without drowning |

> **Why Chip Huyen's *AI Engineering* and Raschka's *Build an LLM from Scratch* are worth buying:** They map *exactly* onto this role — Huyen for the systems/product side (evaluation, RAG, cost, the AI engineering discipline itself), Raschka for the from-scratch internals. Together they cover both competencies (A) and (B) better than any free resource sequence. This is the rare case where paid ≫ free.
>
> **Why learn vLLM specifically:** It's the de facto open serving standard, used or benchmarked against everywhere. Knowing PagedAttention and continuous batching cold is directly interviewable.


---

# Phase 5 — AI Infrastructure & MLOps

**Duration:** 6–8 weeks (heavily overlaps with Phases 4 & 6; much builds on infra you already know). **Objective:** Operationalize AI — the systems, pipelines, and platforms that make models reliable, scalable, observable, and cost-effective in production. **This is where your existing DevOps/distributed-systems background becomes a superpower.**

> **Opinionated stance:** You already know Kubernetes, Docker, CI/CD, and monitoring. Don't relearn them — learn the **AI-specific deltas**: GPU scheduling, model/prompt versioning, drift detection, feature stores, distributed training/inference, and the MLOps tooling landscape. Most "MLOps" is just good software engineering applied to models, which you already do. Focus your time on what's genuinely new.

### Required knowledge
Your existing infra skills + Phases 3–4. Kubernetes/Docker fluency assumed.

### 5.1 Orchestration & compute (the AI deltas)
- **Kubernetes for AI:** GPU scheduling, node pools, the **NVIDIA GPU Operator**, device plugins, MIG (multi-instance GPU), Kueue/Volcano for batch GPU jobs, KServe for model serving.
- **Ray** (learn this — it's the dominant distributed-AI framework): Ray Core, **Ray Serve** (scalable inference), **Ray Train** (distributed training), Ray Data, Ray Tune (HPO). Ray is increasingly the glue for large-scale AI.
- **Docker for AI:** CUDA base images, multi-stage builds for model artifacts, image size management, reproducibility.

### 5.2 Distributed training & inference at scale
- **Distributed training:** DDP, **FSDP** (Fully Sharded Data Parallel), DeepSpeed **ZeRO** (stages 1–3), Megatron-LM (tensor/pipeline parallelism); the 3 parallelisms (data/tensor/pipeline) and when each applies; gradient accumulation, activation checkpointing.
- **Distributed inference:** tensor-parallel serving (vLLM/TensorRT-LLM multi-GPU), model sharding across nodes, disaggregated prefill/decode.
- **GPU memory optimization:** the VRAM budget (params + gradients + optimizer + activations), offloading, quantized training, LoRA to fit big models on small GPUs.

### 5.3 Serving patterns
- **Batch vs streaming vs online** inference; autoscaling (including scale-to-zero for costly GPUs); request queuing/batching; multi-model serving; canary/shadow deployments for models; A/B routing.
- Caching layers: response cache, semantic cache, KV-cache reuse, embedding cache.

### 5.4 The MLOps stack (know the landscape, master one of each)
- **Experiment tracking:** **Weights & Biases** (best-in-class UX) or **MLflow** (open, self-hostable, also does model registry).
- **Orchestration/pipelines:** **Airflow** (mature, ubiquitous), **Dagster** (modern, asset-based, developer-friendly — increasingly preferred), Prefect, **Kubeflow Pipelines** (K8s-native), Metaflow.
- **Model registry & versioning:** MLflow Registry, W&B Artifacts; **data/model versioning** with DVC or LakeFS.
- **Feature stores:** Feast (open), Tecton; when you actually need one (real-time features, training/serving skew prevention) vs when it's overkill.
- **Prompt/model versioning:** treat prompts as versioned code/config; LangSmith/LangFuse/PromptLayer, or git-based prompt management.

### 5.5 Data engineering for AI
- **Data pipelines:** batch + streaming (Kafka), ETL/ELT for training data and RAG ingestion.
- **Lakehouse architecture:** **Spark** / **Databricks**, **Snowflake**, Delta Lake / Iceberg / Hudi; when to use a lakehouse vs warehouse.
- Data quality, validation (Great Expectations/Pandera), lineage.
- *You don't need to become a data engineer — but RAG and training both live or die on data pipelines, so know enough to design them.*

### 5.6 Observability, monitoring & governance (the AI deltas)
- **Model monitoring:** latency/throughput/error rates (you know these) **+** quality metrics, cost-per-request, token usage.
- **Drift detection:** data drift, concept drift, embedding drift, prediction drift; tools (Evidently, Arize, WhyLabs, NannyML).
- **LLM observability:** tracing (LangSmith/LangFuse/Phoenix), token/cost dashboards, eval-in-production, feedback capture.
- **Security & compliance:** model security (prompt injection, data exfiltration via agents, model/data poisoning), PII handling, GDPR/HIPAA/SOC2 considerations, secrets/key management for model APIs, supply-chain security for model weights, the EU AI Act at a high level.

### Phase 5 Deliverables
- [ ] A model deployed on Kubernetes with GPU scheduling + autoscaling.
- [ ] A distributed training run using FSDP or DeepSpeed.
- [ ] An end-to-end pipeline (Dagster/Airflow) that ingests data → trains/updates → evaluates → deploys.
- [ ] A monitoring dashboard with drift detection on a deployed model.
- **Portfolio project:** **"AI Observability Dashboard"** and/or **"Distributed Inference Service"** (Phase 8) — infra projects are *rare* in candidate portfolios and stand out hugely.
- **Interview prep goal:** Design the infra for serving an LLM at scale (autoscaling, batching, cost, multi-region); explain FSDP/ZeRO; design a drift-detection + retraining loop; discuss agent security.

### Phase 5 Curated Resources
| Type | Pick | Cost | Why |
|---|---|---|---|
| **Best overall course** | **Made With ML** (Goku Mohandas) | Free | The best MLOps course — production ML done right, code-first |
| Best complementary course | **Full Stack Deep Learning** (archived but gold) | Free | Best "building ML products" course; still conceptually current |
| Best book | *Designing Machine Learning Systems* (Chip Huyen) | ~$50 | **The** ML systems design book — worth buying |
| Best book (data) | *Designing Data-Intensive Applications* (Kleppmann) | ~$50 | You may already own it; the data-systems bible, fully relevant |
| Best Ray resource | Ray docs + Anyscale blog/tutorials | Free | The source |
| Best distributed-training | HF *"Ultra-Scale Playbook"* + DeepSpeed/PyTorch FSDP docs | Free | HF's playbook is an outstanding, current deep-dive |
| Best MLOps reference | Google Cloud MLOps whitepaper + `awesome-mlops` (GitHub) | Free | Vendor-neutral framework + tool landscape |
| Best observability | Evidently AI docs/blog, Arize courses | Free | Practical drift/monitoring |

> **Why Made With ML over a paid MLOps bootcamp:** It's free, code-first, and genuinely production-grade — built by an engineer who did this at scale. Paid MLOps bootcamps rarely exceed it. **Why *Designing ML Systems* is worth buying:** it's the canonical reference for exactly the system-design questions you'll be asked, and pairs perfectly with Kleppmann's DDIA (which you likely already know from your SWE background).

---

# Phase 6 — AI System Design

**Duration:** 5–6 weeks (interleave with Phase 11 interview prep — they're the same muscle). **Objective:** Learn to design complete AI systems under real constraints (latency, cost, scale, reliability, safety) and *communicate* those designs — because **AI System Design is the highest-signal interview round for Senior AI Engineer roles**, and the skill that defines seniority on the job.

> **Opinionated stance:** This phase is *pattern acquisition*. You already do system design; now you're learning the AI-specific components (retrieval, serving, eval, guardrails, feedback loops) and how they compose. The meta-skill: **always drive to requirements first** (latency budget? scale? accuracy bar? cost ceiling? freshness? privacy?), then design, then discuss tradeoffs. Interviewers grade *how you reason*, not whether you name a specific vendor.

### Required knowledge
Phases 3–5. Your existing system-design skills.

### 6.1 The AI system design framework
For *any* prompt, work through: **Requirements** (functional + non-functional: latency, throughput, cost, accuracy, freshness, privacy, safety) → **Data** (sources, ingestion, quality) → **Model choice** (build/buy, open/closed, size) → **Architecture** (offline/online split, retrieval, serving, caching) → **Evaluation** (offline + online, guardrails) → **Scaling & cost** → **Monitoring & feedback loop** → **Failure modes & mitigations** → **Tradeoffs**.

### 6.2 Canonical systems to be able to design cold
Each below: know the reference architecture, the hard parts, the tradeoffs, and how to evaluate it.

- **Enterprise RAG / knowledge assistant** (permissions-aware retrieval, freshness, citations, multi-tenant).
- **Coding copilot** (context gathering from repo, low-latency completion, FIM, ranking, privacy).
- **Customer support AI** (deflection, escalation, tone/guardrails, knowledge freshness, human handoff).
- **Document intelligence / IDP** (OCR → layout → extraction → validation; structured output; human-in-the-loop).
- **AI search engine** (query understanding, hybrid retrieval, reranking, generative answers with citations, freshness).
- **Recommendation system** (candidate generation → ranking → re-ranking; the classic two-tower + feature store design).
- **Voice agent** (STT → LLM → TTS pipeline, latency budgeting <800ms, barge-in, turn-taking, streaming).
- **Multimodal system** (image+text ingestion, VLM choice, cross-modal retrieval).
- **Autonomous agent platform** (tool registry, planning, memory, guardrails, cost control, observability, human oversight).
- **Knowledge graph + LLM** (GraphRAG, entity extraction, graph construction, hybrid graph+vector retrieval).
- **Evaluation platform** (golden sets, LLM-judge, regression gates, A/B infra, feedback capture).

### 6.3 Cross-cutting design levers
Offline vs online inference · latency optimization (caching, speculative decoding, model routing, prefill/decode split) · caching (response/semantic/KV/embedding) · horizontal scaling & autoscaling · cost optimization (model cascades, batching, quantization, caching) · reliability (fallbacks, timeouts, circuit breakers, graceful degradation) · observability (tracing, evals-in-prod) · security (prompt injection, data isolation, PII, agent sandboxing).

### 6.4 Diagramming
Practice drawing these fast and clearly. Example skeleton (Enterprise RAG):

```
                          ┌─────────────────────────────────────────────┐
                          │                  CLIENTS                     │
                          │        (web / mobile / API / Slack)          │
                          └───────────────────────┬─────────────────────┘
                                                  │  (auth, rate-limit)
                                        ┌─────────▼──────────┐
                                        │    API GATEWAY     │
                                        │  + AuthN/AuthZ     │
                                        └─────────┬──────────┘
                                                  │
                    ┌─────────────────────────────▼──────────────────────────────┐
                    │                     ORCHESTRATION LAYER                     │
                    │   query rewrite → retrieve → rerank → assemble ctx → LLM    │
                    │           guardrails (in/out) · prompt versioning           │
                    └───┬───────────────┬──────────────────┬──────────────┬──────┘
                        │               │                  │              │
              ┌─────────▼───┐   ┌───────▼───────┐   ┌──────▼──────┐  ┌────▼──────────┐
              │  RETRIEVAL  │   │   RERANKER    │   │  LLM SERVING│  │  GUARDRAILS   │
              │ vector (dense)│  │ cross-encoder │   │  (vLLM /    │  │ PII, injection│
              │ + BM25 (sparse)│ │ / Cohere      │   │  API, GPU   │  │ moderation    │
              │  hybrid fusion │ │  rerank       │   │  autoscaled)│  │               │
              └───────┬────────┘ └───────────────┘   └─────────────┘  └───────────────┘
                      │
       ┌──────────────▼───────────────┐        ┌───────────────────────────────────┐
       │      INGESTION PIPELINE       │        │            OBSERVABILITY          │
       │ sources → parse → chunk →      │        │ tracing · token/cost · evals-in-  │
       │ embed → index (perms-aware)    │        │ prod · drift · user feedback      │
       │ (Dagster/Airflow, incremental) │        └───────────────────────────────────┘
       └────────────────────────────────┘
                      │
       ┌──────────────▼───────────────┐        ┌───────────────────────────────────┐
       │   VECTOR DB + METADATA STORE  │        │        OFFLINE EVAL HARNESS       │
       │ (pgvector/Qdrant + Postgres)  │        │ golden set · RAGAS · regression   │
       │  ACL filters, multi-tenant    │        │ gate in CI before prompt/model    │
       └────────────────────────────────┘        └───────────────────────────────────┘
```

### Phase 6 Deliverables
- [ ] A **design doc** for each of 6+ canonical systems (1–2 pages each: requirements, architecture diagram, tradeoffs, eval plan, failure modes). *This doc set is directly reusable in interviews.*
- **Portfolio project:** The **"End-to-End Enterprise AI Platform"** capstone (Phase 8) is your chance to *implement* one of these designs.
- **Interview prep goal:** Given any of the canonical prompts, produce a clear design in 35–45 min with requirements-gathering, a diagram, and explicit tradeoff discussion.

### Phase 6 Curated Resources
| Type | Pick | Cost | Why |
|---|---|---|---|
| Best book | *Designing Machine Learning Systems* (Huyen) + *AI Engineering* (Huyen) | ~$50 ea | Directly on-target; the case studies *are* interview prep |
| Best free case studies | **Evidently AI's ML System Design database** (~450 real case studies) + company eng blogs (Uber, Netflix, DoorDash, LinkedIn, Meta, Pinterest, Instacart) | Free | Real architectures from real scale — read voraciously |
| Best system-design prep | *ByteByteGo* (Alex Xu) + his AI/ML system design content | Free/Paid | You know the general framework; adapt it to AI |
| Best LLM-app patterns | *"What We Learned from a Year of Building with LLMs"* (O'Reilly, applied-llms.org), Anthropic's *"Building Effective Agents"*, Chip Huyen's blog | Free | Battle-tested patterns from practitioners |
| Best RAG/agent design | LlamaIndex + LangChain conceptual docs, Anthropic/OpenAI cookbooks | Free | Reference designs |

> **Why company engineering blogs over generic courses:** Real production AI architectures (with the constraints and compromises intact) teach system design better than any polished course. The Evidently case-study database is a goldmine specifically for this.


---

# Phase 7 — AI Research Literacy

**Duration:** Ongoing from Phase 3 onward (~3–4 hrs/week, not a discrete block). **Objective:** Read papers efficiently, track the frontier without drowning, and understand the lineage of ideas well enough to discuss architecture tradeoffs and reason about *new* models in interviews. **You're building research *literacy*, not becoming a researcher.**

> **Opinionated stance:** You do not need to reproduce papers or derive proofs. You need to (1) read a paper in 30–60 min and extract the core idea, (2) know the ~25 papers that shaped the field, and (3) keep a light finger on the pulse. The lab interviews (OpenAI/Anthropic/DeepMind) *will* probe whether you understand *why* architectures are the way they are — this phase is your defense.

### How to read a paper (the 3-pass method — Keshav)
1. **Pass 1 (5–10 min):** Title, abstract, intro, section headings, figures, conclusion. Answer: what problem, what approach, what result?
2. **Pass 2 (~1 hr):** Read carefully, skip proofs, understand figures/method. Can you explain it to a peer?
3. **Pass 3 (deep, selective):** Only for papers you must implement or truly master — reconstruct the method, question assumptions.
- **Tools:** Use an LLM as a reading copilot (ask it to explain notation, summarize, quiz you). Keep paper notes in your repo (problem / idea / method / result / why it matters / limitations).

### The essential reading list (grouped, with *why*)

**Foundations & Transformers**
- **Attention Is All You Need** (2017) — introduced the transformer; *the* foundational paper. Still 100% relevant — everything descends from it.
- **BERT** (2018) — bidirectional pretraining, the encoder paradigm; matters historically and for embeddings/retrieval; still used for rerankers/encoders.
- **GPT-2 / GPT-3** (2019/2020) — scaling + in-context learning; GPT-3's "few-shot learner" reframed the field. Relevant as the conceptual basis of modern LLMs.
- **Scaling Laws** (Kaplan 2020) & **Chinchilla** (2022) — compute-optimal training; *why* models are sized as they are. Highly relevant.
- **Vision Transformer (ViT)** (2020) — transformers for images; unified architecture story. Relevant.

**Instruction-following & Alignment**
- **InstructGPT** (2022) — the RLHF pipeline that made models usable; *the* bridge from GPT-3 to ChatGPT. Essential.
- **DPO** (2023) — preference tuning without RL; now the dominant alignment method. Very relevant.
- **Constitutional AI** (2022, Anthropic) — RLAIF/self-critique alignment. Relevant, especially for Anthropic.

**Efficiency & Fine-tuning**
- **LoRA** (2021) — low-rank fine-tuning; the basis of nearly all practical fine-tuning. Essential and current.
- **QLoRA** (2023) — 4-bit + LoRA; democratized fine-tuning. Relevant.
- **FlashAttention** (2022, v2 2023) — IO-aware attention; why modern training/inference is fast. Essential to understand.
- **Mixtral / Mixture-of-Experts** (2024) & **Switch Transformer** (2021) — sparse models; frontier-relevant (most SOTA models are MoE).

**Retrieval & Knowledge**
- **RAG** (2020, Lewis et al.) — the original retrieval-augmented generation paper. Essential context.
- **GraphRAG** (2024, Microsoft) — knowledge-graph-augmented retrieval. Relevant for advanced RAG.
- **Dense Passage Retrieval / ColBERT** — retrieval foundations. Useful.

**Multimodal & Generative**
- **CLIP** (2021) — contrastive image-text; the basis of multimodal retrieval and many VLMs. Essential.
- **Whisper** (2022) — robust ASR; the standard for speech. Relevant if you touch voice.
- **Segment Anything (SAM)** (2023) — foundation model for segmentation; promptable vision. Relevant for vision.
- **Diffusion / DDPM** (2020) + **Latent Diffusion / Stable Diffusion** (2022) — image generation; essential if doing generative vision, useful context otherwise.

**Open models & frontier architectures**
- **Llama / Llama 2 / Llama 3** — the open-weights lineage; read the Llama 3 report for a modern training recipe. Highly relevant.
- **DeepSeek-V3 / DeepSeek-R1** (2024–2025) — MoE at scale + RL-based reasoning (GRPO); frontier-defining and interview-hot. Very relevant.
- **Mamba / State Space Models** (2023) — a non-attention sequence architecture; the leading "what might replace transformers" candidate. Relevant for architecture discussions (be able to contrast with attention).

**Agents & tool use**
- **ReAct** (2022) — reasoning + acting; foundational agent pattern. Relevant.
- **Toolformer** (2023) — self-taught tool use. Useful context.
- Anthropic **"Building Effective Agents"** (2024, engineering post, not a paper) — the most practical agent guidance available. Read it.

### Staying current (without drowning)
- **Weekly:** Latent Space, The Batch (DeepLearning.AI), Simon Willison's blog, AK's (@_akhaliq) paper picks, Sebastian Raschka's *Ahead of AI* newsletter.
- **Selective deep-dives:** Hugging Face Daily Papers, arxiv-sanity; pick 1 paper/week to Pass-2.
- **Podcasts:** Latent Space, Dwarkesh (for depth/strategy).

### Phase 7 Deliverables
- [ ] **`paper-notes`** repo with structured summaries of all ~25 essential papers.
- [ ] Reproduce/annotate 2–3 (e.g., LoRA on a small model, a mini-transformer per *Attention Is All You Need*, a tiny diffusion model).
- **Interview prep goal:** For any paper on the list, explain in 3 minutes: the problem, the key idea, why it mattered, and whether it's still SOTA. Contrast transformers vs Mamba, RLHF vs DPO, dense vs MoE.

### Phase 7 Curated Resources
| Type | Pick | Cost | Why |
|---|---|---|---|
| Best paper collections | **The Annotated Transformer** (Harvard) + Karpathy's paper-based videos | Free | Code-annotated papers = deepest understanding |
| Best "explained" blogs | Lilian Weng, Jay Alammar, Sebastian Raschka, Yannic Kilcher (YouTube paper reviews) | Free | The best paper-explainers; use them as your Pass-1.5 |
| Best newsletters | Sebastian Raschka's *Ahead of AI*, The Batch, Latent Space | Free | Curated frontier without the firehose |
| Best method | *"How to Read a Paper"* (Keshav, 3-pass) | Free | The technique |


---

# Phase 8 — Hands-on Portfolio

**Duration:** Continuous — projects are woven through all phases, not done at the end. **Objective:** Build a portfolio of increasingly sophisticated systems that *prove* you can ship production AI. **Your portfolio, not your certificates, gets you interviews.** Quality over quantity: 5–6 deep, well-documented, deployed projects beat 20 demos.

> **Opinionated stance:** Every project must be (1) **deployed** (a live URL or reproducible deploy), (2) **evaluated** (you can prove it works, with metrics), (3) **documented** (a README that reads like a design doc: problem, architecture, tradeoffs, results, what you'd do next), and (4) **on GitHub** with clean commits. The *evaluation* and *writeup* are what separate senior candidates. A RAG demo is worthless; a RAG system with a measured retrieval-quality benchmark and a cost analysis is a hire signal.

### Universal project standards (apply to all)
- **GitHub structure:** `README.md` (design-doc quality) · `src/` · `tests/` · `evals/` · `infra/` (Docker/IaC) · `notebooks/` (exploration) · `docs/` (architecture diagram + decisions) · CI (lint/test/eval gate).
- **Every project:** typed Python, tests, an eval harness, a Dockerfile, a deployment, and a writeup.

### The project ladder (15 projects, escalating difficulty)

| # | Project | Core skills | Difficulty | Time | Resume value |
|---|---|---|---|---|---|
| 1 | **LLM Chatbot with memory & streaming** | LLM APIs, streaming, context mgmt, basic eval | ★☆☆☆☆ | 1 wk | Foundation; shows API fluency |
| 2 | **Tabular ML pipeline (Kaggle)** | Classical ML, CV, leakage prevention, SHAP, W&B | ★★☆☆☆ | 2 wk | Proves evaluation discipline |
| 3 | **Semantic search engine** | Embeddings, vector DB, hybrid search, reranking | ★★☆☆☆ | 1.5 wk | Retrieval fundamentals |
| 4 | **Production RAG over your docs** | Full RAG, chunking, RAGAS eval, citations, guardrails | ★★★☆☆ | 3 wk | **Flagship — the most-asked-about system** |
| 5 | **Resume reviewer / structured extraction** | Structured outputs, function calling, evals, prompt eng | ★★☆☆☆ | 1 wk | Practical, demoable |
| 6 | **OCR / document intelligence pipeline** | Vision+OCR, layout parsing, extraction, validation, HITL | ★★★☆☆ | 2.5 wk | IDP is in high demand |
| 7 | **Vision model (train + deploy)** | CNN/ViT, transfer learning, mixed precision, serving | ★★★☆☆ | 2 wk | Shows DL training chops |
| 8 | **Fine-tuning project (LoRA/QLoRA)** | PEFT, dataset curation, TRL/Axolotl, before/after eval | ★★★★☆ | 3 wk | **Differentiator — proves you can adapt models** |
| 9 | **Voice assistant** | STT (Whisper) → LLM → TTS, latency budgeting, streaming | ★★★☆☆ | 2 wk | Multimodal, latency-sensitive design |
| 10 | **AI coding assistant** | Repo context, retrieval, FIM, ranking, low-latency serving | ★★★★☆ | 3 wk | Directly relevant to many target companies |
| 11 | **Multi-agent workflow** | Agents, tool use, MCP server, orchestration, guardrails | ★★★★☆ | 3 wk | **Frontier — 2026's hottest area** |
| 12 | **AI search engine** | Query understanding, hybrid retrieval, generative answers, citations | ★★★★☆ | 3 wk | Complex retrieval + generation |
| 13 | **Recommendation engine** | Two-tower, candidate gen → ranking, feature store, offline+online eval | ★★★★☆ | 3 wk | Classic + interviewed everywhere |
| 14 | **Distributed inference service** | vLLM, quantization, K8s + GPU autoscaling, load testing, cost/latency benchmark | ★★★★★ | 3 wk | **Rare in portfolios — huge infra signal** |
| 15 | **AI observability dashboard** | Tracing, token/cost, drift detection, eval-in-prod, feedback loop | ★★★★☆ | 2 wk | **Rare — proves production maturity** |

### The capstone

| # | Project | Time | Resume value |
|---|---|---|---|
| 16 | **End-to-End Enterprise AI Platform** — combine RAG + agents + eval + observability + K8s serving + CI/CD into one multi-tenant, permissioned, monitored, cost-controlled platform with a design doc and load tests. | 6–8 wk | **The centerpiece.** Demonstrates every competency at once. This is your "staff-level thinking" proof. |

### How to choose (if time-limited)
Non-negotiables for an LLM-focused Senior role: **#4 (RAG), #8 (Fine-tuning), #11 (Multi-agent), #14 (Distributed inference)**, plus **#16 (Capstone)**. Add #2 (classical ML) if targeting generalist roles, #10/#13 if targeting specific companies (coding tools / recsys). 

### Phase 8 Deliverables & interview value
- [ ] 5–6 polished, deployed, evaluated, documented projects + 1 capstone.
- [ ] Each with a blog-style writeup (post on your site / Medium / dev.to — *searchable credibility*).
- **Interview prep goal:** For each project, be able to give a 5-min architecture walkthrough, defend every design decision, discuss what broke and how you fixed it, and describe how you'd scale/improve it. *These become your behavioral + system-design stories.*

---

# Phase 9 — Open Source Contributions

**Duration:** Ongoing from ~Month 4 (once you have working knowledge). **Objective:** Build credibility, deepen expertise through real codebases, and create a public track record that recruiters and hiring managers *actually check*. **A merged PR to vLLM or Transformers is worth more than most certificates.**

> **Opinionated stance:** OSS contribution is the highest-credibility, lowest-cost signal available — and it's *genuinely* how you learn production AI code. Start with docs/tests/bugs to learn a codebase's norms, then graduate to features. Depth in one project beats scattered one-off PRs. Target projects you *actually use* in your Phase 8 work — your bug reports will be real.

### Where to contribute, by project & entry point

| Project | Good first contributions | Experienced contributions | Why it builds credibility |
|---|---|---|---|
| **Hugging Face Transformers** | Docs, model card fixes, failing tests, `good first issue` label | New model implementations, pipeline features, bug fixes in generation | The most recognized ML OSS org; huge visibility |
| **vLLM** | Docs, benchmarks, small bug fixes | Kernel/scheduler optimizations, new model support, quantization | The serving standard; deep systems cred |
| **LangChain / LangGraph** | Docs, integrations, examples | New integrations, core abstractions, bug fixes | High visibility; easy entry via integrations |
| **LlamaIndex** | Docs, examples, connectors | Retrieval/eval features, integrations | RAG credibility |
| **PyTorch** | Docs, tests, small fixes | Ops, autograd, distributed, performance | **Elite signal** — hard, but maximal credibility |
| **Ray** | Docs, examples, bug triage | Serve/Train/Data features, scaling fixes | Distributed-AI cred |
| **MLflow** | Docs, bug fixes, flavors | Tracking/registry features, integrations | MLOps cred |
| **LiteLLM** | Provider integrations, docs | Routing, caching, proxy features | Easy, high-utility entry point |
| **Ollama / Open WebUI / AnythingLLM** | Docs, bug fixes, UI/features | Model support, backend features | Approachable, popular, good for early wins |
| **PEFT / TRL / Axolotl / Unsloth** | Docs, examples, bug fixes | New PEFT/alignment methods, trainer features | Fine-tuning cred; smaller communities = easier impact |
| **SGLang** | Docs, benchmarks | Kernels, structured-gen features | Cutting-edge serving cred |
| **FastAPI** | Docs, examples, bug fixes | Features (you likely already know it) | Leverages your existing SWE strength |

### The contribution ladder
1. **Learn the norms (Weeks 1–2 per project):** read CONTRIBUTING.md, fix a docs error, add a test, close a `good first issue`.
2. **Fix real bugs:** reproduce, write a failing test, fix, PR with a clean description.
3. **Add features:** propose in an issue first, get buy-in, implement with tests + docs.
4. **Become a recognized contributor:** consistent quality PRs to one project → maintainers know your name → referrals happen.

### How to build credibility (the meta-game)
- Contribute where you have real usage (your Phase 8 projects surface real bugs).
- Write excellent PR descriptions (context, approach, testing, tradeoffs — like a design doc).
- Engage in issues/discussions helpfully before PRing.
- Blog about your contributions ("How I added X to vLLM").
- **Result:** a GitHub profile showing merged PRs to respected AI infra = a standing interview magnet.

### Phase 9 Deliverables
- [ ] 3–5 merged PRs across 1–2 projects you use, escalating from docs → bug → feature.
- **Interview prep goal:** A concrete "tell me about a hard technical contribution" story rooted in real OSS work.

---

# Phase 10 — Certifications

**Duration:** Opportunistic — only pursue certs that clear the value bar. **Objective:** Be honest about what certs do and don't do. **For Senior AI Engineer roles at top companies, certifications are near the bottom of the signal hierarchy** (portfolio > OSS > experience > referrals >> certs). Pursue them only for: (a) cloud credibility if targeting a specific cloud-heavy employer, (b) forcing-function structure if you learn better with an exam goal, or (c) consulting/enterprise contexts where they're contractually valued.

> **Opinionated stance:** Do **not** spend months chasing certificates. No cert will get you a Senior role at OpenAI/Anthropic/DeepMind — your ability to build and reason will. Certs are most useful for cloud-platform signaling and for MLOps/K8s roles. The DeepLearning.AI courses are worth doing for *content*, not for the certificate. Treat this phase as strictly optional.

### Ranked certifications

**MUST HAVE:** *(None are truly "must-have" for top-lab Senior roles. If forced to pick the highest-value one:)*
- **AWS Certified Machine Learning – Specialty** *or* **Google Professional ML Engineer** — only if targeting cloud/enterprise employers who filter on it.

**USEFUL:**
| Cert | Recognition | Interview value | Practical value | Difficulty | Renewal | Verdict |
|---|---|---|---|---|---|---|
| **Google Professional ML Engineer** | High (enterprise) | Low–Med | Med | Med–Hard | 2 yr | Best-regarded ML cloud cert; do it if cloud-signaling matters |
| **AWS ML Specialty** (retiring → **AWS ML Engineer Associate**) | High (enterprise) | Low–Med | Med | Med | 3 yr | Broad AWS shops value it |
| **Azure AI Engineer Associate** | Med–High (MS shops) | Low | Med | Med | 1 yr | Only for Azure-heavy targets |
| **CKA / CKAD** (Kubernetes) | High (infra) | Med (for infra roles) | **High** | Med–Hard | 3 yr | **The most genuinely useful cert here** — real K8s skill, respected, and you'll use it for AI infra |
| **Databricks ML Associate/Professional** | Med–High (data/ML) | Low–Med | Med–High | Med | 2 yr | Valuable if targeting Databricks-ecosystem or lakehouse roles |

**OPTIONAL:**
| Cert | Verdict |
|---|---|
| **DeepLearning.AI specializations** (DL, ML, LLM, etc.) | Do for **content**, ignore the certificate. Excellent learning, negligible credential value. |
| **Hugging Face** courses (free, no formal cert weight) | Do for content — genuinely current and hands-on. |
| **NVIDIA DLI certifications** | Nice for GPU/CUDA/Triton specialization signaling; optional. |
| **Snowflake** certs | Only for Snowflake-ecosystem data roles. |
| **TensorFlow Developer Certificate** | Mostly obsolete for this path (PyTorch-first world). Skip unless an employer names it. |

**AVOID (for this goal):**
- Generic "AI certificates" from unknown providers, pay-to-pass bootcamp certs, anything marketed primarily to beginners, and any cert positioned as a *substitute* for building. They can even slightly *hurt* by signaling a credential-chasing profile to senior interviewers.

### Phase 10 verdict
If you do *any*: **CKA** (real skill + respected) and, only if cloud-signaling matters for your targets, **one** of Google PMLE / AWS ML. Everything else: consume the *content*, skip the *credential*. Reallocate the saved time to Phase 8 (portfolio) — far higher ROI.


---

# Phase 11 — Senior AI Engineer Interview Preparation

**Duration:** 8–10 weeks of focused prep (interleaved with Phase 6), ramping up in the final 2–3 months before applying. **Objective:** Convert your knowledge and portfolio into interview performance across every round type. **Senior AI Engineer loops typically include:** (1) coding, (2) ML/DL fundamentals, (3) LLM depth, (4) ML/AI system design, (5) a project deep-dive, and (6) behavioral — plus, at labs, research discussion.

> **Opinionated stance:** The two rounds that most often decide senior offers are **AI System Design** and the **project deep-dive** — both reward the exact "reason under constraints and communicate tradeoffs" skill this roadmap builds. Don't neglect coding (still gates you), but your edge as an experienced SWE is that you're *already good at it*. Spend disproportionate prep on ML system design, LLM depth, and telling your project stories crisply.

### What each round tests & how to prep

| Round | What they probe | Prep |
|---|---|---|
| **Coding** | DS&A + ML-flavored coding (implement k-means, attention, a data pipeline; sometimes LeetCode-medium) | 3–4 wks LeetCode (focus mediums; arrays/strings/graphs/DP) + implement ML primitives from scratch |
| **ML/DL fundamentals** | Bias–variance, regularization, metrics, backprop, optimizers, overfitting, CNN/RNN/transformer mechanics | Flashcards + explain-out-loud; the 100 ML Qs below |
| **LLM depth** | Attention, KV cache, RAG, fine-tuning vs RAG, quantization, agents, eval, cost/latency | The 100 LLM Qs below; your Phase 4 projects |
| **ML/AI system design** | Design RAG/recsys/copilot/agent platform under constraints | The 50 design Qs + Phase 6 design docs; practice out loud/whiteboard |
| **Project deep-dive** | Depth on YOUR work: decisions, tradeoffs, failures, scale | Rehearse each Phase 8 project as a 5-min + deep Q&A story |
| **Behavioral** | Leadership, ambiguity, conflict, impact (senior bar = influence beyond your own code) | STAR stories; map to Amazon LPs / company values |
| **Research discussion** (labs) | Paper understanding, architecture reasoning, curiosity | Phase 7 paper notes; be able to discuss 5–6 papers deeply |

---

## Coding Questions (50) — DS&A + ML-flavored

**DS&A core (1–25):** two-sum/variants · valid anagram · group anagrams · longest substring without repeat · product of array except self · trapping rain water · merge intervals · insert interval · top-K frequent elements · k closest points · merge k sorted lists · LRU cache · min stack · valid parentheses · binary tree level-order · lowest common ancestor · serialize/deserialize tree · number of islands · course schedule (topo sort) · clone graph · word ladder · coin change · longest increasing subsequence · edit distance · word break.

**ML/AI-flavored implementation (26–50):** implement k-means from scratch · implement k-NN · implement linear/logistic regression with gradient descent · implement a decision-tree split (Gini/entropy) · implement PCA via SVD · implement softmax (numerically stable) · implement cross-entropy loss + gradient · implement scaled dot-product attention · implement multi-head attention · implement a single transformer block · implement layer norm · implement batch norm (train/eval) · implement dropout · implement a basic autograd (scalar) · implement BPE tokenizer (merge loop) · implement cosine similarity + top-k retrieval · implement mini-batch data loader · implement beam search · implement greedy/temperature/top-p sampling · implement a sliding-window chunker · implement reservoir sampling · implement a simple KV-cache · implement embedding-based semantic search · implement reciprocal rank fusion (hybrid search) · implement an eval metric (F1 / NDCG / MRR).

> **Practice sources:** LeetCode (Blind 75 / NeetCode 150 for DS&A) · Karpathy's videos (for the ML primitives — you'll have built most already) · `ml-interview` and `deep-learning-interview` GitHub repos.

---

## ML Questions (100) — fundamentals, classical ML, DL

**Statistics & foundations (1–15):** Explain bias–variance and how to diagnose each · what is overfitting and 5 ways to combat it · precision vs recall — when to optimize which · ROC-AUC vs PR-AUC and when each misleads · how does cross-validation work and when does it fail (time series?) · what is data leakage — give 3 types · explain the curse of dimensionality · MLE vs MAP · what is regularization, L1 vs L2 · why does L1 induce sparsity · confidence interval vs prediction interval · type I vs II error · p-value misconceptions · how to handle class imbalance (5 methods) · calibration — what and why.

**Classical ML (16–45):** How does a decision tree choose splits · random forest vs gradient boosting · XGBoost vs LightGBM vs CatBoost · how does gradient boosting work · bagging vs boosting · SVM and the kernel trick · k-means limitations and alternatives · how to choose k in k-means · DBSCAN vs k-means · PCA — what it does and its assumptions · t-SNE vs UMAP · handling categorical features (encodings) · feature scaling — when needed · feature selection methods · SHAP vs permutation importance · how to detect/handle multicollinearity · naive Bayes assumptions · logistic regression — why "regression" · linear regression assumptions · handling missing data · dealing with outliers · imbalanced regression targets · how to build a recommendation system (two-stage) · collaborative vs content-based filtering · cold-start solutions · evaluation of ranking (NDCG/MAP/MRR) · A/B test design for an ML model · offline vs online metric divergence · concept drift — detect and respond · when a simpler model beats a complex one.

**Deep learning (46–80):** Explain backpropagation · vanishing/exploding gradients — causes and fixes · why ReLU over sigmoid · activation function tradeoffs (ReLU/GELU/SiLU) · weight initialization — why it matters (Xavier/He) · batch norm — what/why/train-vs-eval · layer norm vs batch norm — why LN in transformers · dropout — mechanism and inference behavior · SGD vs Adam vs AdamW · learning-rate schedules (warmup, cosine) · gradient clipping — when/why · what consumes GPU memory in training · mixed precision — how and why bf16 · residual connections — the gradient argument · CNN: convolution, pooling, receptive field · why CNNs for images · RNN vs LSTM vs GRU · why transformers replaced RNNs · self-attention — the math · multi-head attention — why multiple heads · positional encoding — why needed, RoPE · transfer learning — when/how · catastrophic forgetting · data augmentation strategies · label smoothing · how to debug a model that won't converge · how to debug NaN loss · how to debug overfitting vs underfitting · loss functions — MSE vs cross-entropy · softmax numerical stability · embedding — what it represents geometrically · how to choose batch size · effect of learning rate · epochs vs steps · early stopping.

**Applied/production ML (81–100):** How to detect data drift · retraining triggers · training/serving skew — cause and prevention · feature store — when needed · shadow deployment · canary for models · how to monitor an ML model in prod · model versioning strategy · reproducibility in ML · how to evaluate a model before shipping · handling feedback loops · when to use a heuristic vs ML · how to estimate the value/ROI of an ML project · how to scope an ambiguous ML problem · dealing with limited labeled data (active learning, weak supervision) · semi-supervised approaches · handling non-stationarity · cost-sensitive learning · fairness/bias in models — detect and mitigate · how to explain a model to non-technical stakeholders.


---

## LLM Questions (100) — the core of modern AI interviews

**Transformer & architecture internals (1–25):** Walk through self-attention step by step · why scale by √d_k · what is the KV cache and how does its size scale · MQA vs GQA vs MHA — tradeoffs · what is RoPE and why does it help length extrapolation · absolute vs relative positional encoding · what does FlashAttention optimize and how · pre-norm vs post-norm · RMSNorm vs LayerNorm · SwiGLU — why · what is a Mixture-of-Experts model and why use it · dense vs sparse models — FLOPs vs params · how does causal masking work · encoder vs decoder vs encoder-decoder — when each · what limits context length · attention complexity and how to reduce it · what is speculative decoding · prefill vs decode phases · why is decode memory-bound · what are scaling laws (Chinchilla) · tokenization: BPE vs WordPiece vs SentencePiece · why tokenization causes arithmetic/spelling failures · what is a logit, temperature, top-p, top-k · greedy vs beam vs sampling · what is perplexity.

**RAG & retrieval (26–50):** Design a RAG pipeline end to end · chunking strategies and tradeoffs · fixed vs semantic vs recursive chunking · how to choose an embedding model (MTEB) · dense vs sparse retrieval · what is hybrid search and reciprocal rank fusion · why add a reranker — cross-encoder vs bi-encoder · when do you NOT need a vector DB · pgvector vs Qdrant vs Pinecone tradeoffs · how to evaluate RAG (RAGAS: faithfulness, relevance) · retrieval metrics (recall@k, MRR, NDCG) · query rewriting / HyDE / multi-query · parent-document / contextual retrieval · what is GraphRAG and when to use it · how to handle stale/updated documents · permission-aware retrieval in enterprise RAG · handling tables/images in RAG · reducing hallucination in RAG · citation/grounding strategies · chunk size vs retrieval quality tradeoff · embedding fine-tuning — when worth it · handling long documents · multi-vector retrieval (ColBERT) · reducing RAG latency · RAG vs long-context — when each.

**Fine-tuning & alignment (51–70):** When to fine-tune vs RAG vs prompt · what does fine-tuning change (behavior vs knowledge) · explain LoRA and its math · QLoRA — how 4-bit + LoRA works · full FT vs PEFT tradeoffs · catastrophic forgetting in fine-tuning · how to build a fine-tuning dataset · SFT vs preference tuning · explain RLHF (reward model + PPO) · explain DPO and why it's simpler than RLHF · DPO vs PPO vs GRPO · what is Constitutional AI / RLAIF · reward hacking · how to evaluate a fine-tuned model · data quality vs quantity in fine-tuning · instruction tuning · what is distillation · how to fine-tune for structured output · overfitting in fine-tuning · how much data do you need to fine-tune.

**Serving, inference & optimization (71–85):** How would you cut LLM inference cost 5× · what is continuous/dynamic batching · what is PagedAttention · vLLM vs TGI vs TensorRT-LLM · quantization: GPTQ vs AWQ vs GGUF · INT4/INT8/FP8 tradeoffs · TTFT vs TPOT — how to optimize each · how to load-test an LLM endpoint · model routing / cascades for cost · prompt caching and semantic caching · how to serve multiple models efficiently · autoscaling GPU inference (scale-to-zero) · latency budget for a real-time LLM feature · throughput vs latency tradeoff · estimating tokens/sec and cost per request.

**Agents, tools & applications (86–100):** Design an agent with tool use · ReAct vs plan-and-execute · when do multi-agent systems help vs hurt · what is MCP and why does it matter · agent memory: short vs long-term · how to make agents reliable (tool-call accuracy, guardrails) · prompt injection — attack and defense · how to evaluate an agent · function/tool calling — how it works · structured output / constrained decoding · handling agent failures and loops · cost control for agentic systems · human-in-the-loop design · LLM-as-judge and its biases · how to evaluate an LLM feature you can't fully specify.

> **Practice sources:** your Phase 4 projects (the best prep) · Lilian Weng's blog (agents, hallucination, prompting) · Chip Huyen's *AI Engineering* · the `LLM-Interview` GitHub repos · Hugging Face course quizzes.

---

## System Design Questions (50)

**LLM/GenAI systems (1–25):** Design enterprise RAG for 10M internal docs with permissions · design a coding copilot (like Copilot/Cursor) · design a customer-support AI with escalation · design a document-intelligence pipeline (invoices → structured data) · design an AI search engine with generative answers · design a voice agent under 800ms latency · design a multimodal (image+text) assistant · design an autonomous-agent platform with tool registry · design a knowledge-graph + LLM system · design an LLM evaluation platform · design a semantic-caching layer for an LLM API · design a content-moderation system using LLMs · design a real-time translation service · design a meeting-summarization product · design a personalized email-writing assistant · design a code-review bot · design a chatbot that stays factually current · design a multi-tenant LLM SaaS (isolation, cost attribution) · design an LLM gateway/router across providers · design a prompt-management/versioning system · design a system to detect/reduce hallucinations · design an LLM-powered analytics (text-to-SQL) system · design a long-document Q&A system · design a fine-tuning pipeline as a service · design guardrails for an agent with database access.

**Classical ML / infra systems (26–50):** Design a recommendation system (feed ranking) · design a news-feed ranking system · design an ad click-through-rate prediction system · design a fraud-detection system · design a spam/abuse classifier at scale · design a search-ranking system · design a real-time personalization system · design an anomaly-detection pipeline · design a feature store · design an ML training platform · design a model-serving platform (multi-model, autoscaled) · design a distributed training system for large models · design an experiment-tracking/model-registry system · design a data pipeline for ML (batch + streaming) · design a drift-detection + auto-retraining system · design an A/B testing platform for ML · design a real-time bidding system · design an image-similarity search at scale · design a video-recommendation system · design a demand-forecasting system · design an ML monitoring/observability platform · design a data-labeling pipeline · design a batch-inference system for billions of records · design a GPU-cluster scheduler · design an end-to-end MLOps platform.

**How to answer (the rubric interviewers use):** (1) clarify requirements & constraints (2–3 min) → (2) define scope & success metrics → (3) high-level architecture (diagram) → (4) deep-dive 2–3 components → (5) evaluation strategy → (6) scaling & cost → (7) failure modes, monitoring, tradeoffs. **Always state assumptions; always discuss tradeoffs; never name-drop a tool without justifying it.**

---

## Take-Home Projects (20 representative prompts)

1. Build a RAG system over a provided corpus; report retrieval + answer quality metrics.
2. Fine-tune a small open model on a given task; show before/after evals.
3. Build an agent that completes a multi-step task with tools; measure success rate.
4. Optimize a given LLM endpoint for cost/latency; document the improvement.
5. Build a text classifier; justify your model choice and evaluation.
6. Build a semantic search API with hybrid retrieval + reranking.
7. Implement an eval harness (LLM-as-judge) for a given use case.
8. Build a structured-extraction pipeline with validation.
9. Deploy a model to a container with autoscaling; load-test it.
10. Build a drift-detection monitor for a provided model + data stream.
11. Implement a transformer block from scratch; train on a toy task.
12. Build a recommendation system on a provided dataset; report offline metrics.
13. Build a document-QA system with citations; measure faithfulness.
14. Build a multi-provider LLM gateway with routing + caching.
15. Implement quantization on a model; compare quality/speed/memory.
16. Build a guardrails layer (PII + injection defense) for an LLM app.
17. Build a data pipeline that ingests + chunks + embeds + indexes documents.
18. Create a prompt-versioning + regression-testing workflow.
19. Build a small voice pipeline (STT → LLM → TTS) and measure latency.
20. Build an observability dashboard for an LLM app (tokens, cost, traces, feedback).

> These map directly onto your Phase 8 projects — *if you build the portfolio, you've pre-completed most take-homes.*

## Mock Interview Scenarios (20)

1. 45-min system design: enterprise RAG with permissions. 2. LLM depth grill: attention → KV cache → serving optimization. 3. ML fundamentals rapid-fire: bias–variance → metrics → leakage. 4. Coding: implement multi-head attention, then optimize. 5. Project deep-dive: defend your fine-tuning project's every decision. 6. Design a coding copilot end-to-end. 7. Debug scenario: "our RAG returns irrelevant results — diagnose." 8. Debug scenario: "training loss is NaN — walk me through it." 9. Fine-tune vs RAG vs prompt: given a business case, decide + defend. 10. Cost crisis: "our LLM bill is $200k/mo — cut it in half." 11. Agent design: reliable multi-agent workflow with guardrails. 12. Behavioral: a time you influenced a technical decision across teams. 13. Behavioral: a project that failed — what happened and what you learned. 14. Research discussion: explain DPO vs RLHF and when you'd use each. 15. Research discussion: could Mamba replace transformers? Argue both sides. 16. Design an LLM evaluation platform for a product team. 17. Design a recsys feed-ranking system with a feature store. 18. Latency challenge: voice agent under 800ms — design the pipeline. 19. Security scenario: an agent has DB access — threat-model it. 20. Ambiguity: "make our support bot better" — scope, design, measure.

> **How to run mocks:** Pramp / interviewing.io (free/paid peer mocks) · trade mocks with peers · record yourself · use an LLM as an interviewer ("act as a Senior AI Engineer interviewer, ask me a system design question and grade my answer"). **Aim for 15–20 mocks before real loops.**

### Phase 11 Deliverables
- [ ] Completed question banks (able to answer ~90% cold).
- [ ] 6–8 polished STAR behavioral stories mapped to seniority signals.
- [ ] 6+ reusable system-design templates (from Phase 6).
- [ ] 15–20 mock interviews done.
- **Interview prep goal:** Consistent strong performance across all 6 round types; a crisp 5-min pitch for every portfolio project.


---

# Phase 12 — Weekly Learning Schedule

**A realistic 12–18 month plan at ~12–18 hrs/week** (compress to 12 months at 20+ hrs/week; extend to 18 at ~10 hrs/week). The structure repeats weekly: **Theory/Reading (25%) · Coding/Implementation (35%) · Project work (30%) · Paper reading + Interview practice + Revision (10%).**

### The weekly template (every week, all phases)
- **Mon–Tue (theory + reading):** current phase's videos/course/book (~3–4 hrs).
- **Wed–Thu (coding/implementation):** implement the week's concepts; work on the active project (~4–5 hrs).
- **Fri (project + paper):** advance the active portfolio project; Pass-2 read one paper, add to `paper-notes` (~3 hrs).
- **Sat (project deep work):** biggest project block (~3–4 hrs).
- **Sun (revision + interview practice, from Month 4):** spaced-repetition flashcards, one mock or question set, write a short blog note (~2 hrs).

### Month-by-month milestones

**Months 1–2 — Foundations (Phases 0–1, start 2)**
- Wk 1: Phase 0 gap analysis + environment setup + GPU access. Start Karpathy micrograd (preview).
- Wk 2–4: Linear algebra + calculus (3B1B + MML), implement PCA/SVD/backprop in NumPy.
- Wk 5–6: Probability + statistics (Stat110/StatQuest), implement MLE/naive Bayes/bias-variance sim.
- Wk 7–8: Optimization + information theory, implement SGD/Adam/entropy/KL. Begin Andrew Ng ML.
- **Milestone:** `ml-math-from-scratch` repo done; can explain bias–variance, cross-entropy, SVD, Adam verbally.

**Months 2–4 — Classical ML (Phase 2)**
- Wk 9–12: Supervised/unsupervised, trees, boosting (XGBoost/LGBM/CatBoost), sklearn, evaluation, leakage. *Hands-On ML* (Géron).
- Wk 13–16: Recsys, feature engineering, SHAP, experiment tracking. **Project 2 (Kaggle tabular)** — aim top 25% + writeup. **Project 1 (chatbot)** as a warm-up.
- **Milestone:** Kaggle project + writeup published; can choose model+metric+validation for any tabular problem.

**Months 4–7 — Deep Learning (Phase 3)** *(start OSS + interview flashcards)*
- Wk 17–20: Karpathy Zero-to-Hero (micrograd → makemore → nanoGPT), fast.ai in parallel. Build NN fundamentals from scratch.
- Wk 21–24: CNNs/RNNs/attention, PyTorch mastery, mixed precision, DDP. d2l.ai.
- Wk 25–28: **Project 7 (vision, train+deploy)** + **Project 3 (semantic search)**. Transformer from scratch. Start reading foundational papers (Attention, BERT, GPT).
- **Milestone:** Transformer-from-scratch repo + blog; DDP training run; first OSS docs PR.

**Months 7–11 — LLMs (Phase 4)** *(the big one)*
- Wk 29–32: Transformer internals deep (KV cache, RoPE, FlashAttention, MoE), tokenization (build BPE). Raschka *Build an LLM from Scratch*.
- Wk 33–36: RAG deep — **Project 4 (Production RAG)** with hybrid search, reranking, RAGAS. Chip Huyen *AI Engineering*.
- Wk 37–40: Serving + optimization (vLLM, quantization) — **Project 14 (Distributed inference)**. Agents + MCP — **Project 11 (Multi-agent)**.
- Wk 41–44: Fine-tuning + alignment — **Project 8 (LoRA/QLoRA fine-tune)**. Evaluation harnesses. Papers: InstructGPT, LoRA, DPO, RAG, FlashAttention.
- **Milestone:** RAG + agent + fine-tune + serving projects done; can design RAG on a whiteboard and cut inference cost 5×.

**Months 11–14 — Infra + System Design (Phases 5–6)** *(ramp interview prep)*
- Wk 45–48: MLOps stack (W&B/MLflow, Dagster/Airflow, Ray), K8s+GPU, FSDP/DeepSpeed. Made With ML + FSDP playbook.
- Wk 49–52: **Project 15 (Observability dashboard)**; monitoring/drift/security. Data eng (Spark/lakehouse) as needed.
- Wk 53–56: AI System Design — write 6+ design docs; study company eng blogs + Evidently case studies. Design questions from Phase 11.
- **Milestone:** 6+ system-design templates; observability project; can design any canonical AI system.

**Months 14–18 — Capstone + Interview Prep + Applications (Phases 6, 8, 11)**
- Wk 57–62: **Project 16 (Enterprise AI Platform capstone)** — the centerpiece. Continue OSS (aim for a feature PR).
- Wk 63–68: Full interview mode — question banks, 15–20 mocks, behavioral stories, project pitches, coding refresh (NeetCode). Polish portfolio + writeups.
- Wk 69–72: Applications, referrals (from OSS relationships), real loops. Iterate on feedback.
- **Milestone:** Capstone shipped; ~90% question-bank fluency; 15+ mocks; applying with a strong portfolio + OSS record.

> **Adjust to your target:** Lab roles (OpenAI/Anthropic/DeepMind) → add depth to Phases 4D + 7 (alignment, papers, from-scratch). Product/infra roles (Databricks/Snowflake/most startups) → weight Phases 5–6 + Projects 4/11/14/16. Applied science → keep Phase 2 strong.

---

# Final Deliverables

## 1. Dependency Graph

```
                                    ┌──────────────────────┐
                                    │  PHASE 0: Gap Analysis│
                                    │   + GPU/env setup     │
                                    └───────────┬───────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │ PHASE 1: Math Founds. │
                                    │ LinAlg·Calc·Prob·Stats│
                                    │  Optim·InfoTheory     │
                                    └─────┬─────────────┬───┘
                                          │             │
                        ┌─────────────────▼──┐      ┌───▼────────────────────┐
                        │ PHASE 2: Classical  │      │  (math feeds directly  │
                        │ ML  (trees, boosting│      │   into DL & LLMs)       │
                        │  eval, leakage)     │      └───┬────────────────────┘
                        └──────────┬──────────┘          │
                                   │                     │
                                   └──────────┬──────────┘
                                              │
                                  ┌───────────▼────────────┐
                                  │  PHASE 3: Deep Learning │
                                  │  NN from scratch·PyTorch│
                                  │  CNN·RNN·ATTENTION·dist │◄─────┐
                                  └───────────┬────────────┘      │
                                              │                   │
                                  ┌───────────▼────────────┐      │
                                  │  PHASE 4: LLMs          │      │ PHASE 7:
                                  │  internals·RAG·agents·  │◄─────┤ Research
                                  │  serving·fine-tune·eval │      │ Literacy
                                  └──┬──────────────────┬───┘      │ (parallel,
                                     │                  │          │  Mo.4→end)
                     ┌───────────────▼──┐         ┌─────▼──────────┴──┐
                     │ PHASE 5: AI Infra│         │ PHASE 6: AI System│
                     │ MLOps·Ray·K8s·   │◄───────►│ Design            │
                     │ dist·serving·mon │         │ (uses 3+4+5)      │
                     └────────┬─────────┘         └─────────┬─────────┘
                              │                             │
                              └──────────────┬──────────────┘
                                             │
                    ┌────────────────────────▼────────────────────────┐
                    │   PHASE 8: PORTFOLIO (continuous, spans 2→end)   │
                    │   15 projects + capstone (each phase feeds one)  │
                    └────────────────────────┬────────────────────────┘
                                             │
                    ┌────────────────────────▼────────────────────────┐
                    │   PHASE 9: OSS (Mo.4→end) │ PHASE 10: Certs (opt) │
                    └────────────────────────┬────────────────────────┘
                                             │
                                  ┌──────────▼───────────┐
                                  │ PHASE 11: Interview  │
                                  │ Prep (ramps Mo.11→18)│
                                  └──────────┬───────────┘
                                             │
                                  ┌──────────▼───────────┐
                                  │   SENIOR AI ENGINEER  │
                                  │   OFFERS 🎯           │
                                  └──────────────────────┘

  PHASE 12 (weekly schedule) = the time-management layer wrapping ALL phases.
  Legend:  ─►/◄─ = "feeds into / depends on"   ◄──► = "co-developed / mutually reinforcing"
```

## 2. Roadmap Table

| Phase | Focus | Duration | Prerequisites | Key Outcome |
|---|---|---|---|---|
| **0** | Gap analysis + setup | 2–3 days | SWE background | Honest baseline; GPU/env ready |
| **1** | Math foundations | 4–6 wks (compressible) | NumPy | Fluent math intuition; from-scratch repo |
| **2** | Classical ML | 5–7 wks | Phase 1 | Evaluation discipline; Kaggle project |
| **3** | Deep learning | 8–10 wks | Phase 1–2 | PyTorch mastery; transformer from scratch |
| **4** | LLMs | 12–16 wks | Phase 3 | Architect/serve/fine-tune/eval LLM systems |
| **5** | AI infra + MLOps | 6–8 wks (overlaps 4/6) | Phase 3–4 + SWE infra | Production ML systems; distributed train/serve |
| **6** | AI system design | 5–6 wks (overlaps 11) | Phase 3–5 | Design any AI system under constraints |
| **7** | Research literacy | Ongoing (Mo.4→end) | Phase 3 | Read papers; discuss frontier architectures |
| **8** | Portfolio | Continuous (Mo.2→end) | Each phase | 5–6 deployed projects + capstone |
| **9** | OSS | Ongoing (Mo.4→end) | Phase 3–4 | Merged PRs; public credibility |
| **10** | Certifications | Opportunistic (optional) | Varies | (Optional) CKA / one cloud ML cert |
| **11** | Interview prep | 8–10 wks (ramps Mo.11→18) | Phases 2–7 + portfolio | Offer-ready across all rounds |
| **12** | Weekly schedule | Wraps everything | — | Sustainable execution |


## 3. 300+ Task Checklist

> Check these off as you go. Grouped by phase. (300+ discrete tasks.)

### Phase 0 — Gap Analysis & Setup (8)
- [ ] 1. Complete honest self-assessment against the gap table
- [ ] 2. Set up GPU access (Colab Pro+ or Lambda/RunPod/Vast)
- [ ] 3. Install `uv` and configure Python environment
- [ ] 4. Create Hugging Face, W&B, and GitHub learning accounts
- [ ] 5. Create the `learning-journal` repo structure
- [ ] 6. Define your target role tier (lab vs product/infra vs applied)
- [ ] 7. Set your weekly hour budget and calendar blocks
- [ ] 8. Bookmark the core resource set

### Phase 1 — Math Foundations (30)
- [ ] 9. Watch 3B1B Essence of Linear Algebra (full)
- [ ] 10. Watch 3B1B Essence of Calculus (full)
- [ ] 11. Read MML book Part I (linear algebra)
- [ ] 12. Implement matrix multiplication in NumPy
- [ ] 13. Implement PCA from scratch via SVD
- [ ] 14. Implement eigendecomposition of a covariance matrix
- [ ] 15. Implement low-rank image compression with SVD
- [ ] 16. Reproduce a LoRA-style low-rank decomposition by hand
- [ ] 17. Implement backpropagation for a 2-layer MLP in NumPy
- [ ] 18. Verify gradients with finite differences
- [ ] 19. Compute and visualize Jacobians for a vector function
- [ ] 20. Watch Harvard Stat 110 core lectures
- [ ] 21. Implement MLE for a Gaussian
- [ ] 22. Sample from 5 distributions and verify moments empirically
- [ ] 23. Build a naive Bayes classifier from scratch
- [ ] 24. Watch StatQuest bias–variance, ROC/AUC, cross-validation
- [ ] 25. Bootstrap a confidence interval
- [ ] 26. Run a t-test manually and interpret it
- [ ] 27. Simulate the bias–variance tradeoff (polynomial degree)
- [ ] 28. Read Ruder's gradient-descent overview
- [ ] 29. Implement SGD from scratch
- [ ] 30. Implement Momentum from scratch
- [ ] 31. Implement Adam from scratch
- [ ] 32. Visualize optimizer trajectories on a 2D loss surface
- [ ] 33. Reproduce warmup + cosine LR schedule
- [ ] 34. Read Olah's Visual Information Theory
- [ ] 35. Compute entropy, cross-entropy, KL in NumPy
- [ ] 36. Derive why softmax + cross-entropy gives clean gradients
- [ ] 37. Compute perplexity of a toy language model
- [ ] 38. Publish `ml-math-from-scratch` repo + notebook series

### Phase 2 — Classical ML (32)
- [ ] 39. Complete Andrew Ng ML Specialization (audit)
- [ ] 40. Read Hands-On ML (Géron) Part I
- [ ] 41. Derive & implement linear regression (gradient descent)
- [ ] 42. Derive & implement logistic regression
- [ ] 43. Implement a decision-tree split (Gini + entropy)
- [ ] 44. Train & tune a random forest
- [ ] 45. Train & tune XGBoost on a real dataset
- [ ] 46. Train & tune LightGBM; compare to XGBoost
- [ ] 47. Train CatBoost with categorical features
- [ ] 48. Implement k-means from scratch
- [ ] 49. Apply DBSCAN and compare to k-means
- [ ] 50. Apply PCA + t-SNE + UMAP to a dataset
- [ ] 51. Build an isolation-forest anomaly detector
- [ ] 52. Implement k-fold and stratified CV
- [ ] 53. Implement time-series CV correctly
- [ ] 54. Construct and detect 3 types of data leakage
- [ ] 55. Compute precision/recall/F1/ROC-AUC/PR-AUC and interpret
- [ ] 56. Generate SHAP explanations for a model
- [ ] 57. Compute permutation feature importance
- [ ] 58. Handle class imbalance 3 ways; compare
- [ ] 59. Calibrate a classifier and plot reliability
- [ ] 60. Build a collaborative-filtering recommender (MovieLens)
- [ ] 61. Build a content-based recommender
- [ ] 62. Implement candidate-gen → ranking two-stage recsys
- [ ] 63. Compute NDCG/MAP/MRR for rankings
- [ ] 64. Set up W&B experiment tracking
- [ ] 65. Design an A/B test for a model
- [ ] 66. Simulate concept drift and detect it
- [ ] 67. Enter a Kaggle competition
- [ ] 68. Reach top 25% on a Kaggle competition
- [ ] 69. Write a methodology writeup (evaluation + leakage)
- [ ] 70. Publish the tabular ML pipeline project

### Phase 3 — Deep Learning (40)
- [ ] 71. Build micrograd with Karpathy
- [ ] 72. Build makemore (bigram → MLP) with Karpathy
- [ ] 73. Build nanoGPT with Karpathy
- [ ] 74. Complete fast.ai Practical Deep Learning (Part 1)
- [ ] 75. Implement an MLP from scratch (forward + backward)
- [ ] 76. Implement ReLU/GELU/SiLU and compare
- [ ] 77. Implement Xavier & He initialization; measure effect
- [ ] 78. Implement dropout (train/eval behavior)
- [ ] 79. Implement batch norm (train/eval)
- [ ] 80. Implement layer norm
- [ ] 81. Implement a residual block; show gradient-flow benefit
- [ ] 82. Overfit a single batch (debugging technique)
- [ ] 83. Implement LR finder
- [ ] 84. Implement gradient clipping
- [ ] 85. Build a CNN and train on CIFAR-10
- [ ] 86. Apply transfer learning with a pretrained backbone
- [ ] 87. Implement an RNN from scratch
- [ ] 88. Implement an LSTM; observe gradient stability
- [ ] 89. Implement scaled dot-product attention
- [ ] 90. Implement multi-head attention
- [ ] 91. Implement sinusoidal positional encoding
- [ ] 92. Build a full transformer block from scratch
- [ ] 93. Train a char-level transformer on a corpus
- [ ] 94. Master PyTorch `nn.Module` + custom `Dataset`/`DataLoader`
- [ ] 95. Implement a full training/eval loop with checkpointing
- [ ] 96. Enable mixed-precision (bf16) training
- [ ] 97. Profile a model with the PyTorch profiler
- [ ] 98. Identify a data-loading vs compute bottleneck
- [ ] 99. Write a DDP training script
- [ ] 100. Run DDP on 2+ GPUs
- [ ] 101. Compute the VRAM budget for a training run by hand
- [ ] 102. Learn CUDA basics (what a kernel is; why fused kernels)
- [ ] 103. Read d2l.ai attention + transformer chapters
- [ ] 104. Train an image classifier with augmentation + W&B
- [ ] 105. Deploy the vision model behind an API
- [ ] 106. Publish "Transformer From Scratch" repo + blog
- [ ] 107. Read Attention Is All You Need (Pass 2)
- [ ] 108. Read BERT paper (Pass 2)
- [ ] 109. Read GPT-2/GPT-3 papers (Pass 1–2)
- [ ] 110. Add all three to `paper-notes`

### Phase 4 — LLMs (55)
- [ ] 111. Read Raschka Build-an-LLM-from-Scratch (full)
- [ ] 112. Build a BPE tokenizer from scratch
- [ ] 113. Implement byte-level BPE; compare vocab sizes
- [ ] 114. Explain & diagram the KV cache; compute its size
- [ ] 115. Implement a simple KV cache
- [ ] 116. Implement MQA and GQA variants
- [ ] 117. Implement RoPE positional encoding
- [ ] 118. Study FlashAttention; explain the IO-aware insight
- [ ] 119. Implement RMSNorm and SwiGLU
- [ ] 120. Study MoE routing; explain params-vs-FLOPs decoupling
- [ ] 121. Study scaling laws (Chinchilla); summarize implications
- [ ] 122. Complete Hugging Face LLM Course
- [ ] 123. Complete Hugging Face Agents Course
- [ ] 124. Read Chip Huyen AI Engineering (full)
- [ ] 125. Build a semantic search engine (embeddings + vector DB)
- [ ] 126. Choose an embedding model using MTEB; justify it
- [ ] 127. Implement hybrid search (dense + BM25 + RRF)
- [ ] 128. Add a cross-encoder reranker; measure improvement
- [ ] 129. Implement 3 chunking strategies; compare retrieval quality
- [ ] 130. Build a production RAG system end-to-end
- [ ] 131. Add citations/grounding to RAG outputs
- [ ] 132. Set up a RAGAS eval harness (faithfulness, relevance)
- [ ] 133. Compute retrieval metrics (recall@k, MRR, NDCG)
- [ ] 134. Implement query rewriting / HyDE
- [ ] 135. Build a GraphRAG variant; compare to vanilla RAG
- [ ] 136. Implement permission-aware retrieval
- [ ] 137. Add PII detection + redaction guardrails
- [ ] 138. Add prompt-injection defenses
- [ ] 139. Implement structured outputs (JSON mode / Instructor)
- [ ] 140. Implement function/tool calling
- [ ] 141. Build a ReAct agent with tools
- [ ] 142. Build an MCP server
- [ ] 143. Connect an agent to the MCP server
- [ ] 144. Implement agent memory (short + long-term)
- [ ] 145. Build a multi-agent workflow; evaluate reliability
- [ ] 146. Compare single-agent vs multi-agent on a task
- [ ] 147. Serve an LLM with vLLM
- [ ] 148. Measure TTFT and TPOT; load-test the endpoint
- [ ] 149. Quantize a model (GPTQ or AWQ); compare quality/speed/memory
- [ ] 150. Export a model to GGUF; run via llama.cpp/Ollama
- [ ] 151. Export a model to ONNX
- [ ] 152. Build a cost model for an LLM feature
- [ ] 153. Implement model routing/cascade for cost
- [ ] 154. Implement semantic caching
- [ ] 155. Curate a fine-tuning dataset
- [ ] 156. Run a LoRA fine-tune (TRL/Axolotl/Unsloth)
- [ ] 157. Run a QLoRA fine-tune on a larger model
- [ ] 158. Evaluate fine-tuned model before/after
- [ ] 159. Implement a DPO preference-tuning run
- [ ] 160. Study RLHF pipeline; summarize reward-model + PPO
- [ ] 161. Build an LLM-as-judge eval pipeline
- [ ] 162. Identify LLM-judge biases in your harness
- [ ] 163. Benchmark a model on a public benchmark subset
- [ ] 164. Read InstructGPT, LoRA, DPO, RAG, FlashAttention (Pass 2)
- [ ] 165. Publish RAG + agent + fine-tune projects with writeups

### Phase 5 — AI Infra & MLOps (35)
- [ ] 166. Complete Made With ML course
- [ ] 167. Read Designing ML Systems (Huyen)
- [ ] 168. Deploy a model on Kubernetes
- [ ] 169. Configure GPU scheduling (NVIDIA GPU Operator)
- [ ] 170. Set up autoscaling (incl. scale-to-zero) for inference
- [ ] 171. Learn Ray Core basics
- [ ] 172. Deploy inference with Ray Serve
- [ ] 173. Run distributed training with Ray Train
- [ ] 174. Run FSDP distributed training
- [ ] 175. Run a DeepSpeed ZeRO training job
- [ ] 176. Explain the 3 parallelisms (data/tensor/pipeline)
- [ ] 177. Implement gradient accumulation + activation checkpointing
- [ ] 178. Set up MLflow tracking + model registry
- [ ] 179. Build a pipeline in Dagster (or Airflow)
- [ ] 180. Version data/models with DVC
- [ ] 181. Stand up a Feast feature store (evaluate need)
- [ ] 182. Implement prompt versioning
- [ ] 183. Build an ETL pipeline for RAG ingestion
- [ ] 184. Process data with Spark (or Databricks)
- [ ] 185. Set up a lakehouse table (Delta/Iceberg)
- [ ] 186. Add data validation (Great Expectations/Pandera)
- [ ] 187. Instrument LLM tracing (LangFuse/Phoenix)
- [ ] 188. Build a token/cost dashboard
- [ ] 189. Implement drift detection (Evidently)
- [ ] 190. Set up model-quality monitoring in prod
- [ ] 191. Design a retraining trigger
- [ ] 192. Threat-model an agent with tool access
- [ ] 193. Implement secrets/key management for model APIs
- [ ] 194. Review GDPR/HIPAA/SOC2 considerations for AI
- [ ] 195. Read the HF Ultra-Scale Playbook
- [ ] 196. Build the AI observability dashboard project
- [ ] 197. Build a distributed inference service (multi-GPU)
- [ ] 198. Load-test the inference service; document cost/latency
- [ ] 199. Containerize with a CUDA base image (multi-stage)
- [ ] 200. Publish both infra projects with writeups

### Phase 6 — AI System Design (16)
- [ ] 201. Learn the AI system-design framework
- [ ] 202. Design doc: Enterprise RAG (with permissions)
- [ ] 203. Design doc: Coding copilot
- [ ] 204. Design doc: Customer-support AI
- [ ] 205. Design doc: Document intelligence pipeline
- [ ] 206. Design doc: AI search engine
- [ ] 207. Design doc: Recommendation system
- [ ] 208. Design doc: Voice agent (<800ms)
- [ ] 209. Design doc: Autonomous-agent platform
- [ ] 210. Design doc: Evaluation platform
- [ ] 211. Study 20+ company engineering blog architectures
- [ ] 212. Study 20+ Evidently ML system-design case studies
- [ ] 213. Practice drawing 5 architectures fast (whiteboard)
- [ ] 214. Read Anthropic "Building Effective Agents"
- [ ] 215. Read "What We Learned from a Year of Building with LLMs"
- [ ] 216. Convert design docs into reusable interview templates

### Phase 7 — Research Literacy (18)
- [ ] 217. Learn the 3-pass paper-reading method
- [ ] 218. Set up `paper-notes` repo template
- [ ] 219. Read Attention Is All You Need + annotate
- [ ] 220. Read BERT + note
- [ ] 221. Read GPT-2/GPT-3 + note
- [ ] 222. Read Scaling Laws + Chinchilla + note
- [ ] 223. Read InstructGPT + note
- [ ] 224. Read DPO + note
- [ ] 225. Read Constitutional AI + note
- [ ] 226. Read LoRA + QLoRA + note
- [ ] 227. Read FlashAttention + note
- [ ] 228. Read Mixtral/MoE + note
- [ ] 229. Read RAG + GraphRAG + note
- [ ] 230. Read CLIP + note
- [ ] 231. Read Whisper + SAM + note
- [ ] 232. Read Diffusion/DDPM + Latent Diffusion + note
- [ ] 233. Read Llama 3 report + DeepSeek-V3/R1 + Mamba + ReAct + notes
- [ ] 234. Subscribe to 3 newsletters; sustain 1 paper/week

### Phase 8 — Portfolio (16)
- [ ] 235. Project 1: LLM chatbot (memory + streaming)
- [ ] 236. Project 2: Tabular ML pipeline (Kaggle)
- [ ] 237. Project 3: Semantic search engine
- [ ] 238. Project 4: Production RAG (flagship)
- [ ] 239. Project 5: Resume reviewer / structured extraction
- [ ] 240. Project 6: OCR / document intelligence
- [ ] 241. Project 7: Vision model (train + deploy)
- [ ] 242. Project 8: Fine-tuning (LoRA/QLoRA)
- [ ] 243. Project 9: Voice assistant
- [ ] 244. Project 10: AI coding assistant
- [ ] 245. Project 11: Multi-agent workflow
- [ ] 246. Project 12: AI search engine
- [ ] 247. Project 13: Recommendation engine
- [ ] 248. Project 14: Distributed inference service
- [ ] 249. Project 15: AI observability dashboard
- [ ] 250. Project 16: Enterprise AI Platform (capstone)

### Phase 9 — Open Source (10)
- [ ] 251. Choose 1–2 target projects you actually use
- [ ] 252. Read their CONTRIBUTING guides
- [ ] 253. Submit a docs-fix PR
- [ ] 254. Add a missing test PR
- [ ] 255. Close a `good first issue`
- [ ] 256. Reproduce + report a real bug
- [ ] 257. Submit a bug-fix PR with a failing test
- [ ] 258. Propose a feature in an issue
- [ ] 259. Land a feature PR (with tests + docs)
- [ ] 260. Blog about a contribution

### Phase 10 — Certifications (5, optional)
- [ ] 261. Decide if any cert clears the value bar for your targets
- [ ] 262. (If infra-focused) Earn CKA
- [ ] 263. (If cloud-signaling) Earn Google PMLE or AWS ML
- [ ] 264. Consume DeepLearning.AI course content (skip certs)
- [ ] 265. Consume Hugging Face course content

### Phase 11 — Interview Prep (35)
- [ ] 266. Complete NeetCode 150 (DS&A)
- [ ] 267. Implement all 25 ML-flavored coding problems
- [ ] 268. Answer all 100 ML questions (self-graded)
- [ ] 269. Answer all 100 LLM questions (self-graded)
- [ ] 270. Practice all 50 system-design questions (outline each)
- [ ] 271. Write full solutions to 10 system-design questions
- [ ] 272. Build spaced-repetition flashcard deck (Anki)
- [ ] 273. Draft 8 STAR behavioral stories
- [ ] 274. Map stories to seniority signals + company values
- [ ] 275. Rehearse 5-min pitch for each portfolio project
- [ ] 276. Prepare project deep-dive Q&A (decisions/failures/scale)
- [ ] 277. Practice explaining 6 papers in 3 min each
- [ ] 278. Complete 5 peer mock interviews
- [ ] 279. Complete 5 more mocks (system design focus)
- [ ] 280. Complete 5 more mocks (LLM depth focus)
- [ ] 281. Complete 5 more mocks (behavioral + project)
- [ ] 282. Do 5 LLM-as-interviewer sessions
- [ ] 283. Practice the RAG-debugging scenario
- [ ] 284. Practice the NaN-loss debugging scenario
- [ ] 285. Practice the cost-cutting scenario
- [ ] 286. Practice the fine-tune-vs-RAG decision scenario
- [ ] 287. Practice the agent-security threat-model scenario
- [ ] 288. Practice a whiteboard multi-head attention implementation
- [ ] 289. Practice a whiteboard RAG design
- [ ] 290. Practice a whiteboard recsys design
- [ ] 291. Refresh coding (timed mediums) weekly
- [ ] 292. Record yourself answering; review delivery
- [ ] 293. Polish GitHub profile + pin best repos
- [ ] 294. Write blog posts for top 3 projects
- [ ] 295. Prepare a portfolio one-pager / personal site
- [ ] 296. Build target-company list + tailor prep to each
- [ ] 297. Activate referrals (from OSS + network)
- [ ] 298. Complete a full-loop dry run (all 6 rounds in a day)
- [ ] 299. Apply to first-wave companies
- [ ] 300. Iterate prep based on real-loop feedback

### Ongoing habits (10)
- [ ] 301. Maintain the learning journal weekly
- [ ] 302. Ship one blog note per week (from Month 4)
- [ ] 303. Read one paper per week (from Month 4)
- [ ] 304. Do spaced repetition 2×/week (from Month 4)
- [ ] 305. Contribute to OSS monthly (from Month 4)
- [ ] 306. Track experiments in W&B for every project
- [ ] 307. Write tests + an eval harness for every project
- [ ] 308. Deploy every project (live or reproducible)
- [ ] 309. Review the frontier weekly (newsletters/podcasts)
- [ ] 310. Re-run the gap analysis quarterly and adjust


## 4. Portfolio Roadmap (when to build each project)

| Month | Project(s) | Ties to Phase |
|---|---|---|
| Mo 2–3 | P1 Chatbot (warm-up), P2 Kaggle tabular | Phase 2 |
| Mo 4–5 | P3 Semantic search | Phase 3/4 bridge |
| Mo 5–6 | P7 Vision model, "Transformer from scratch" | Phase 3 |
| Mo 7–8 | P4 Production RAG (flagship), P5 Resume reviewer | Phase 4 |
| Mo 8–9 | P6 OCR/IDP, P11 Multi-agent + MCP | Phase 4 |
| Mo 9–10 | P8 Fine-tuning, P14 Distributed inference | Phase 4/5 |
| Mo 10–11 | P9 Voice, P10 Coding assistant, P12 AI search, P13 Recsys *(pick per target)* | Phase 4/6 |
| Mo 11–12 | P15 Observability dashboard | Phase 5 |
| Mo 13–15 | **P16 Enterprise AI Platform (capstone)** | Phase 6/8 |

**Minimum viable portfolio (if time-constrained):** P4 (RAG) → P8 (fine-tune) → P11 (multi-agent) → P14 (distributed inference) → P16 (capstone). Add P2 for generalist roles.

## 5. Reading Roadmap (books + papers, sequenced)

**Books — in order of when to read:**
1. *Mathematics for Machine Learning* (free) — Phase 1
2. *Hands-On Machine Learning*, Géron (buy) — Phase 2
3. *Introduction to Statistical Learning* (ISLP, free) — Phase 2 reference
4. *Machine Learning with PyTorch and Scikit-Learn*, Raschka (buy) — Phase 3
5. *Dive into Deep Learning* (d2l.ai, free) — Phase 3 reference
6. *Build a Large Language Model (From Scratch)*, Raschka (buy) — Phase 4
7. *AI Engineering*, Chip Huyen (buy) — Phase 4 **(highest priority for this role)**
8. *Hands-On Large Language Models*, Alammar & Grootendorst (buy) — Phase 4
9. *Designing Machine Learning Systems*, Huyen (buy) — Phase 5/6
10. *Designing Data-Intensive Applications*, Kleppmann (buy; you may own it) — Phase 5 reference
11. *Deep Learning*, Goodfellow (free) — deep reference, selective

**Papers — sequenced (all in Phase 7, mapped to phases):**
- *During Phase 3:* Attention Is All You Need → BERT → GPT-2/3 → ViT
- *During Phase 4:* Scaling Laws → Chinchilla → InstructGPT → RAG → LoRA → QLoRA → FlashAttention → DPO → Mixtral/MoE → Constitutional AI → GraphRAG
- *During Phase 4/7 (multimodal/frontier):* CLIP → Whisper → SAM → Diffusion/DDPM → Latent Diffusion → Llama 3 → DeepSeek-V3/R1 → Mamba → ReAct → Toolformer

## 6. Interview Roadmap

| Timeframe | Focus |
|---|---|
| Mo 4+ (background) | Build Anki deck; add cards as you learn; weekly light coding |
| Mo 11–12 | Start question banks (ML + LLM); write first system-design solutions |
| Mo 13–14 | System-design intensive; behavioral stories; project pitches |
| Mo 15–16 | Mocks (15–20); full-loop dry run; coding refresh; polish portfolio |
| Mo 16–18 | Applications, referrals, real loops, feedback iteration |

**Round-by-round readiness checklist:**
- [ ] Coding: NeetCode 150 done + ML primitives implementable cold
- [ ] ML/DL: 100 ML questions answerable verbally
- [ ] LLM: 100 LLM questions answerable verbally
- [ ] System design: 6+ reusable templates + 10 full solutions
- [ ] Project deep-dive: 5-min pitch + deep Q&A per project
- [ ] Behavioral: 8 STAR stories mapped to seniority + values
- [ ] Research (labs): 6 papers discussable in depth

## 7. Certification Roadmap

| Priority | Cert | When | Only if… |
|---|---|---|---|
| Highest-value (still optional) | **CKA** | Mo 11–13 (leverages Phase 5) | Targeting infra-heavy roles |
| Situational | **Google PMLE** *or* **AWS ML** | Mo 12–14 | Target employer filters on cloud certs |
| Situational | **Databricks ML** | Mo 12–14 | Targeting lakehouse/Databricks ecosystem |
| Content-only | DeepLearning.AI / Hugging Face | As you learn | Always (for content, ignore credential) |
| Skip | TensorFlow Dev, generic AI certs | — | An employer explicitly names it |

**Bottom line:** For top-lab Senior roles, invest ~0 net time in certs beyond CKA-if-relevant. Portfolio + OSS + interview performance dominate.

## 8. Top 100 AI Engineering Skills (2026)

### Core ML (1–10)
1. Supervised/unsupervised learning fundamentals · 2. Bias–variance reasoning · 3. Proper cross-validation (incl. time-series) · 4. Data-leakage detection & prevention · 5. Evaluation metrics selection (ROC-AUC/PR-AUC/F1/NDCG) · 6. Gradient boosting (XGBoost/LightGBM/CatBoost) · 7. Feature engineering & importance (SHAP) · 8. Class-imbalance handling · 9. Recommendation systems (two-stage) · 10. Experiment tracking discipline.

### Deep Learning (11–25)
11. Backpropagation & autograd · 12. PyTorch mastery · 13. Training-loop engineering & debugging · 14. Regularization (dropout/weight decay/augmentation) · 15. Normalization (BatchNorm/LayerNorm/RMSNorm) · 16. Optimizers (SGD/Adam/AdamW) & LR schedules · 17. Residual/skip connections · 18. CNNs & transfer learning · 19. Sequence models (RNN/LSTM/GRU) · 20. Attention & multi-head attention · 21. Transformer architecture · 22. Embeddings · 23. Mixed-precision training · 24. GPU memory reasoning · 25. Profiling & bottleneck analysis.

### LLMs (26–45)
26. Transformer internals (attention/FFN/norm) · 27. KV cache mechanics · 28. MQA/GQA · 29. RoPE & positional encodings · 30. FlashAttention understanding · 31. MoE architectures · 32. Tokenization (BPE/SentencePiece) · 33. Sampling & decoding (temp/top-p/beam) · 34. Prompt & context engineering · 35. Structured outputs & function calling · 36. RAG pipeline design · 37. Chunking strategies · 38. Embedding-model selection · 39. Hybrid search + reranking · 40. RAG evaluation (RAGAS) · 41. Agent design (ReAct/tools) · 42. MCP · 43. LoRA/QLoRA/PEFT fine-tuning · 44. Preference tuning (DPO/RLHF concepts) · 45. LLM evaluation (LLM-as-judge, benchmarks).

### AI Infrastructure (46–57)
46. LLM serving (vLLM/TensorRT-LLM) · 47. Continuous batching & PagedAttention · 48. Quantization (GPTQ/AWQ/GGUF) · 49. Distributed training (FSDP/DeepSpeed/ZeRO) · 50. Distributed/tensor-parallel inference · 51. GPU scheduling on Kubernetes · 52. Ray (Serve/Train/Data) · 53. Autoscaling (incl. scale-to-zero) · 54. Caching (response/semantic/KV/embedding) · 55. Model serving patterns (batch/stream/online) · 56. Inference cost & latency optimization · 57. Edge/local deployment (llama.cpp/Ollama).

### MLOps (58–67)
58. Experiment tracking (W&B/MLflow) · 59. Model registry & versioning · 60. Pipeline orchestration (Dagster/Airflow/Kubeflow) · 61. Data/model versioning (DVC) · 62. Feature stores (Feast) · 63. Prompt versioning · 64. CI/CD for ML (eval gates) · 65. Drift detection · 66. Model monitoring & observability (LLM tracing) · 67. Reproducibility.

### Data Engineering (68–75)
68. Data pipelines (batch + streaming/Kafka) · 69. Spark / distributed processing · 70. Lakehouse (Databricks/Delta/Iceberg) · 71. Warehouse (Snowflake) · 72. RAG ingestion pipelines · 73. Data quality & validation · 74. Data lineage · 75. ETL/ELT design.

### AI System Design (76–83)
76. Requirements/constraint elicitation · 77. Architecture design & diagramming · 78. Offline vs online inference design · 79. Evaluation-pipeline design · 80. Reliability (fallbacks/degradation) · 81. Multi-tenancy & isolation · 82. Feedback-loop design · 83. Tradeoff analysis & communication.

### Software Engineering (84–88)
84. Production Python (typing/testing) · 85. API design (FastAPI) · 86. Testing & eval harnesses · 87. Containerization (Docker/CUDA images) · 88. Clean architecture & code review.

### Cloud (89–92)
89. Cloud AI platforms (AWS/GCP/Azure ML) · 90. Managed GPU compute · 91. Cost management/FinOps for AI · 92. IaC (Terraform).

### Distributed Systems (93–96)
93. Parallelism strategies (data/tensor/pipeline) · 94. Sharding & partitioning · 95. Message queues & async processing · 96. Consistency/latency tradeoffs at scale.

### Security (97–99)
97. Prompt-injection & jailbreak defense · 98. PII handling & data isolation · 99. Agent sandboxing & threat modeling.

### Research Literacy (100)
100. Reading papers & tracking SOTA to reason about new architectures.

## 9. Why These Choices (over alternatives)

**Karpathy Zero-to-Hero > Coursera DL Specialization (for Phase 3 start):** You stated a goal of reasoning about *new* architectures. That requires building autograd and transformers by hand, which Karpathy forces and Coursera abstracts away. Ng's course is excellent for structure/certificate but insufficient for depth.

**PyTorch > TensorFlow:** Research, most industry, and nearly every company on your target list are PyTorch-first. TF/Keras is legacy in this context; learn it only on employer demand. This focuses your limited time.

**Chip Huyen's *AI Engineering* + Raschka's *Build an LLM from Scratch* (paid) > free-only:** They map exactly onto the two competencies of this role (systems + internals) more coherently than any free sequence. This is the rare paid ≫ free case; the cost is trivial versus the time saved and depth gained.

**Hands-On ML (Géron) > Bishop's PRML (for Phase 2):** Géron teaches you to *ship*; Bishop teaches you to *derive*. You are an engineer, not a researcher. Keep Bishop as optional deep-reference.

**Made With ML > paid MLOps bootcamps:** Free, code-first, genuinely production-grade, built by a practitioner who did this at scale. Paid bootcamps rarely exceed it. Reallocate saved money to GPU compute.

**vLLM as the serving default:** It's the industry-standard open serving stack; PagedAttention/continuous batching are directly interviewable and broadly transferable. Learning it generalizes better than learning a single vendor's proprietary stack.

**Company engineering blogs + Evidently case studies > generic system-design courses:** Real production architectures (with real constraints and compromises) teach AI system design better than polished but sanitized courses. This is where seniority is actually forged.

**Portfolio + OSS ≫ certifications:** For Senior roles at top companies, the signal hierarchy is portfolio > OSS > experience > referrals >> certs. Time spent building and contributing has far higher ROI than credential-chasing. Certs earn a place only for cloud-signaling or K8s (CKA), and even then optionally.

**DPO-forward alignment coverage:** DPO now dominates practical preference tuning; RLHF is taught conceptually because you must understand the lineage, but you'll rarely run PPO yourself. This weighting matches real industry practice in 2026.

**RAG + evaluation weighted heaviest in Phase 4:** These are the most-built and most-interviewed LLM competencies, and evaluation is the single most underrated senior differentiator. The roadmap deliberately over-invests here relative to, say, from-scratch pretraining, which almost no engineer does on the job.

---

## Final note on execution

This roadmap is deliberately **maximal** — it covers everything a Senior AI Engineer might be expected to know across the full spectrum of target companies. **You will not do all of it, and that's correct.** Use the critical path (Phase 1 light → 3 → 4 → 5/6 → 11) and the minimum-viable portfolio as your spine; add breadth based on your specific target roles. The three things that will actually get you hired, in order: **(1) a portfolio of deployed, evaluated, well-documented AI systems, (2) the ability to reason about tradeoffs in system-design and project deep-dive rounds, and (3) genuine depth in LLM systems (RAG, agents, serving, fine-tuning, evaluation).** Everything else supports those three. Build relentlessly, evaluate rigorously, write about what you learn, and contribute in public. That combination — on top of the software engineering you already have — is what makes an elite Senior AI Engineer.

*Good luck. Now go build.*
