# Software validation

The following CPU software checks were reported in the v0.1.0 package validation record and are retained in v0.1.1:

- 16 automated test cases passed (including all five model variants with finite forward/backward gradients).
- A synthetic dataset was created locally; one epoch of training, checkpoint saving/reloading, three-step evaluation and per-sample/summary export completed end to end.
- The same synthetic predictions were passed through the external comparator scorer; aggregate rotation/translation errors matched the internal evaluation.
- Both qualitative projection and ablation plotting commands produced readable image files.
- The full paper configuration produced RGB tokens of shape [1,480,128] and three 4x4 poses for one sample.

Recorded test runtime: Python 3.12, PyTorch 2.15.0+cpu. Synthetic fixtures and resulting weights/metrics are excluded from the release archive. These checks establish software functionality only; they do not constitute a numerical reproduction of the manuscript's KITTI or private-platform quantitative results. Version 0.1.1 is a documentation and metadata update; it does not report new benchmark runs.
