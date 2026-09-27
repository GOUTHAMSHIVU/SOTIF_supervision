# ADR 0001: Record Architecture Decisions Using Repository-Managed ADRs

* **Status:** Accepted
* **Date:** 2026-09-27
* **Author:** Goutham S

## Context and Problem Statement
The development of an ISO 21448 (SOTIF) perception supervisor involves complex architectural choices, safety constraints, hardware trade-offs, and mathematical compromises. We need a centralized, immutable method to document engineering rationale so that context is preserved for the dissertation, safety audits, and future team members without getting lost in chat transcripts or external notebooks.

## Decision Drivers
* Need for version-controlled traceability alongside source code (Git-native).
* Compliance with ISO 26262 / ISO 21448 documentation and justification standards.
* Prevention of design drift and recurring debates over settled architectural choices.
* Compatibility with local markdown viewers and Obsidian vaults.

## Considered Options
* Option 1: Git-tracked Markdown Architecture Decision Records (MADR format) in `docs/adr/`
* Option 2: External cloud wikis (Confluence, Notion)
* Option 3: Inline code comments and docstrings

## Decision Outcome
Chosen option: **Option 1 (Git-tracked Markdown Architecture Decision Records)**, because:
* Every architectural choice is version-controlled, diffable, and reviewed via pull requests.
* Documentation remains accessible offline alongside training scripts and models.
* Content can be directly transposed into dissertation chapters and functional safety technical manuals.

## Consequences
* **Positive:** Architectural intent is locked and auditable; clear rationale preserved for academic defense and safety review.
* **Negative:** Requires team discipline to draft and update an ADR when major design shifts occur.