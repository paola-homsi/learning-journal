# Roadmap Progress Tracker

Source: [The Senior AI Engineer Roadmap](./ROADMAP.md) — the full 310-task checklist, tracked here.
Update weekly during the Sunday review. Quarterly: re-run the Phase 0 gap analysis.


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


