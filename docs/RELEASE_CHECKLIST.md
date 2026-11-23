# Reproducibility checklist

Use this checklist when preparing independent training or evaluation runs:

- Use the original drive/frame assignments and fixed evaluation perturbations for exact manuscript reproduction.
- Keep the configuration, software versions, run logs, manifest hashes and checkpoints associated with each experiment.
- Use the same test frames, perturbations, and metric definitions for every compared method.
- Train the ablation variants independently; the constant-gate control retains the valid-count indicator and learns b.
- Generate numerical summaries directly from evaluated predictions.
- Run the automated tests and synthetic smoke test in the intended software environment.
- Keep third-party implementations separate and respect their licenses.
- Obtain private-platform data from the corresponding author upon reasonable request; do not redistribute private records without authorization.
