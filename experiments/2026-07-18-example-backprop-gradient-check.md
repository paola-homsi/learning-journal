# Experiment: Backprop from scratch + finite-difference gradient check  *(EXAMPLE — replace with your own)*

**Date:** 2026-07-18
**Related project:** ml-math-from-scratch (roadmap tasks #17-18)
**W&B run:** n/a (NumPy only)
**Code:** notebooks/backprop_mlp.ipynb @ a1b2c3d

## Hypothesis / question
Can I derive and implement the backward pass of a 2-layer MLP by hand, and do my analytic gradients match numerical gradients to <1e-6?

## Setup
2-layer MLP (784→64→10), ReLU, softmax + cross-entropy, batch of 32 MNIST digits, float64 for the check.

## What I did
Derived dL/dW2, dL/db2, dL/dW1, dL/db1 on paper via chain rule; implemented forward + backward in NumPy; compared against central finite differences (ε=1e-5) on 50 random parameters per tensor.

## Results
| Metric | Target | Got |
|---|---|---|
| Max relative gradient error | < 1e-6 | 3.2e-8 |
| Loss after 500 steps (SGD lr=0.1) | decreasing | 2.30 → 0.41 |

## What broke / surprised me
- First attempt: gradient error ~1e-2 on W1. Bug: I applied the ReLU mask to the wrong pre-activation (used a2's mask instead of a1's). Lesson: name variables by layer, always.
- Softmax overflowed in float32 until I subtracted the row max — "numerically stable softmax" is now visceral, not trivia.
- Surprise: the combined softmax+cross-entropy gradient really is just (p - y). Deriving it once made the "clean gradients" claim obvious.

## Conclusion & next step
Analytic backprop verified. Next: extend the same check to a layer-norm layer (its backward is famously fiddly) — task #80.
