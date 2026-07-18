# Transformers — flashcards  *(starter examples; grow this yourself)*

Q: Why scale attention scores by √d_k?
A: Dot products grow with dimension; large values saturate softmax → tiny gradients. Scaling keeps scores in a well-conditioned range.

Q: What exactly does the KV cache store, and how does its size scale?
A: The K and V tensors of every past token at every layer, so decode doesn't recompute them. Size ≈ layers × kv_heads × head_dim × seq_len × 2 × bytes/dtype — linear in sequence length, and the reason GQA/MQA exist.

Q: Why did LayerNorm (not BatchNorm) win in transformers?
A: BatchNorm's statistics depend on the batch — unstable for variable-length sequences, small batches, and autoregressive decode. LayerNorm normalizes per token, batch-independent.
