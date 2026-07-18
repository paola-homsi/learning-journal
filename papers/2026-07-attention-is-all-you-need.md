# Attention Is All You Need  *(EXAMPLE NOTE — replace with your own reading)*

**Authors / Year:** Vaswani et al., 2017 (Google)
**Link:** arXiv:1706.03762
**Read:** 2026-07-18 · **Pass:** 2
**Status of the idea in 2026:** Historical foundation — and still the architecture underneath everything.

## One-sentence summary
Replaces recurrence entirely with self-attention, making sequence modeling parallelizable and unlocking the scaling era.

## The problem it solves
RNNs/LSTMs process tokens sequentially — training can't parallelize across the sequence, and long-range dependencies degrade through many recurrent steps.

## The key idea
Every token attends directly to every other token via scaled dot-product attention: Attention(Q,K,V) = softmax(QKᵀ/√d_k)V. Direct pairwise connections replace the sequential bottleneck; position is injected separately (sinusoidal encodings) since attention itself is order-blind.

## How it works (Pass 2)
- Encoder-decoder, 6 layers each; each layer = multi-head attention + FFN, with residuals + LayerNorm.
- Multi-head: h=8 parallel attention "views" in subspaces of d_model/h, concatenated — lets heads specialize (syntax, coreference, position...).
- √d_k scaling keeps dot products from saturating softmax at large dimensions.
- Decoder uses causal masking (can't attend to future tokens) + cross-attention to encoder output.

## Results that matter
SOTA on WMT'14 En-De/En-Fr translation at a fraction of prior training cost — the cost reduction (parallelism) mattered more than the BLEU gain.

## Why it mattered
Parallelism → scaling laws became exploitable → BERT, GPT, and everything since. Modern LLMs are decoder-only descendants with the attention core intact (plus RoPE, RMSNorm, SwiGLU, GQA refinements).

## Limitations & open questions
O(n²) attention cost in sequence length — the root cause of today's long-context problem; FlashAttention optimizes constants, SSMs (Mamba) attack the asymptotics. Original sinusoidal encodings extrapolate poorly → RoPE/ALiBi.

## My 3-minute interview explanation
"Before 2017, sequence models were recurrent — inherently serial, so you couldn't parallelize training across tokens, and gradients degraded over long ranges. The transformer's move was to delete recurrence and let every token attend directly to every other token through query-key-value attention, with multiple heads attending in parallel subspaces. Two consequences: training parallelizes fully across the sequence, and any two positions are one step apart. That's what made scale practical — and scale is the story of the next decade. The costs: quadratic attention in sequence length, and no inherent notion of order, so position gets injected separately. Almost every refinement since — RoPE, GQA, FlashAttention, MoE — is engineering around this exact design's bottlenecks while keeping its core."

## Connections
- → papers/flashattention (attacks the memory/IO cost)
- → papers/mamba (attacks the O(n²) asymptotics)
- → experiments: my from-scratch transformer (roadmap task #92-93)
