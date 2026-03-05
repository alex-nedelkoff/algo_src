# Paper Summary Template

Use this template when creating a new paper summary in `../research-vault/papers/`.

---

```markdown
---
title: "Full Paper Title"
authors: [First Author, Second Author, ...]
year: 2025
source: arXiv:XXXX.XXXXX | Conference Name
type: paper
one_line: "Concrete claim with quantitative result, e.g. 'Achieves 95% gate detection at 20m/s using lightweight CNN'"
relevance: 4  # 1-5, how directly applicable to our drone racing stack
tags: [perception, gate-detection, cnn]  # from controlled vocabulary
date_added: 2026-XX-XX
---

## TL;DR

3-5 sentences covering:
- What problem does this solve?
- What is the core approach?
- What is the key result?
- What is the main takeaway for our project?

## Key Findings

- Finding 1: one sentence, quantitative where possible (e.g., "93.2% mAP on gate detection at 640x480, 45 FPS on Jetson Orin")
- Finding 2: ...
- Finding 3: ...
- (max 7 bullets)

## Method Summary

2-3 paragraphs covering:
- Architecture and key design choices
- Training procedure and data
- Evaluation methodology

Enough detail to decide whether to replicate or adapt the approach.

## Relevance to Our Work

1-2 paragraphs explicitly connecting to our stack:
- Which module does this affect? (perception / control / state_estimation / sim)
- What could we adopt directly?
- What would need adaptation for our hardware (Jetson Orin NX, monocular camera)?

## Technical Details

Detailed content for implementation reference:
- Network architecture specifics (layer counts, dimensions, activations)
- Training hyperparameters (learning rate, batch size, epochs, optimizer)
- Ablation results and what they tell us
- Mathematical formulations where relevant
- Hardware/latency benchmarks

## Open Questions

- What remains unclear or untested?
- What follow-up experiments would we need?

## References

- Links to related vault documents: `[related-paper](../papers/slug.md)`
- External links: [arXiv](https://arxiv.org/abs/XXXX.XXXXX), [code](https://github.com/...)
```
