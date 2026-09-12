---
title: EduPulse API
emoji: 🎓
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Early-warning risk scoring for students, with explanations
---

# EduPulse API

The FastAPI service behind [EduPulse](https://github.com/ZainDev04/edupulse): early-warning risk scoring from background
attributes, math-score prediction with conformal intervals, SHAP explanations, a fairness audit with per-group
thresholds, and drift monitoring.

Interactive docs: `/docs`. Health: `/health`.

This Space ships the trained models from the repository; nothing is trained at build time. Source, tests, model cards
and the web front end live on GitHub.
