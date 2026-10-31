# Agent instructions

<!-- egm-por-defecto -->
## Design with EGM (Elephant-Goldfish Model) by default

Before implementing any change that needs design decisions (a new feature, service or component; a change to the
architecture, data model, API or infrastructure), use the EGM plugin (`google/elephant-goldfish-model`; in Claude
Code the commands take the `/egm:` prefix):

1. `egm-brainstorm <idea>`: explore requirements and trade-offs; produces `concepts_brief.md`.
2. `egm-draft`: write the spec section by section (`design_draft.md`) with the owner's feedback.
3. `egm-review <draft>` (or `egm-review-loop`), then `egm-resolve` on the findings the owner picks.
4. Implement only after the owner approves the design. If the code drifts from an approved design: `egm-drift-report`.

Not needed for small, clear fixes (a scoped bug, copy, a config tweak, an operational task). When unsure, ask in one
line. Design documents live in `docs/design/<topic>/` and are versioned with the code; no secrets or personal data.
