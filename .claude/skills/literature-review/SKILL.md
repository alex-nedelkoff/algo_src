# Literature Review

Add paper summaries and concept notes to the research vault (`../research-vault/`).

## When to Use

Trigger when the user:
- Mentions a paper, arXiv link, DOI, or asks to "summarize" research
- Asks to "add to vault", "document this concept", or "review this paper"
- Wants to understand a technique relevant to drone racing

## Workflows

### Paper Summary
1. Fetch/read the paper (arXiv, PDF, or user-provided content)
2. Read the template: `references/paper-template.md`
3. Write summary to `../research-vault/papers/<slug>.md` using the template
4. Update `../research-vault/index/topics.md` with a new row in the appropriate section

### Concept Note
1. Gather information about the concept
2. Read the template: `references/concept-template.md`
3. Write note to `../research-vault/concepts/<slug>.md` using the template
4. Update `../research-vault/index/topics.md` with a new row in the appropriate section

## Quality Rules

- `one_line` must contain a concrete claim or quantitative result — not "explores X" but "achieves 95% gate detection at 20m/s"
- Key Findings must be bullet points, not prose paragraphs
- Relevance section must name a specific module: `perception`, `control`, `state_estimation`, or `sim`
- Tags use controlled vocabulary: `perception`, `control`, `state-estimation`, `sim-to-real`, `domain-randomization`, `gate-detection`, `vio`, `ekf`, `ppo`, `cnn`, `tensorrt`
- Slug format: `first-author-year-keyword` for papers, `descriptive-name` for concepts

## Templates

Full templates are in `references/` — read them when writing a new document:
- `references/paper-template.md` — paper summary with progressive disclosure layers
- `references/concept-template.md` — domain concept note
