# Concept Note Template

Use this template when creating a new concept note in `../research-vault/concepts/`.

---

```markdown
---
title: "Concept Name"
type: concept
one_line: "Concrete summary, e.g. 'EKF fuses IMU + visual odometry for 6-DOF pose at 200Hz with <1cm drift per meter'"
tags: [state-estimation, ekf]  # from controlled vocabulary
date_added: 2026-XX-XX
---

## Definition

2-3 sentences in plain language. What is this concept? A new teammate should understand the gist.

## Why It Matters

1 paragraph connecting this concept to our drone racing project. Be specific about which module (perception, control, state_estimation, sim) and what problem it solves.

## How It Works

2-4 paragraphs at a joining-teammate level:
- Core mechanism or algorithm
- Key assumptions and constraints
- Typical inputs and outputs
- Computational considerations (especially for Jetson Orin NX)

## Variants and Tradeoffs

| Variant | Pros | Cons | Our Fit |
|---------|------|------|---------|
| Variant A | ... | ... | ... |
| Variant B | ... | ... | ... |

## Our Approach

What we chose (or plan to choose) and why. Reference specific code paths in `algo_src/` where applicable:
- Implementation location: `algo_src/module/...`
- Key design decisions and rationale

## Further Reading

- Vault papers: `[related-paper](../papers/slug.md)`
- External: [resource name](https://...)
```
