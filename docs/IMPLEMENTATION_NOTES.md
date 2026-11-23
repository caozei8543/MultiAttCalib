# MultiAttCalib implementation details

This document describes the architecture, mathematical conventions and experimental settings of the MultiAttCalib implementation included in this submission package. The full experiment configuration is in [`configs/paper.json`](../configs/paper.json).

| Component | Implementation in this package |
|---|---|
| Dual ResNet-18 and RGB CCAM-inspired RCB fusion | Independently initialized ResNet-18 branches; project stride 8/16/32 RGB features to the token grid, apply residual depthwise/pointwise blocks, concatenate and merge |
| 480 tokens, 128 channels | 12 x 40 grid, learned positional encodings, four-head residual self-attention and LayerNorm, independent branches |
| Pose head | Concatenate image and depth tokens, reshape to the spatial grid, apply three convolutional layers and adaptive average pooling to obtain six regression outputs; the model converts these to a bounded twist using tanh and per-component bounds |
| Sparse projection and valid-point loss | Floor rasterization, nearest positive camera depth for coincident pixels, depth normalized by 80 m; fixed reference point set uses positive depth and in-image projection without explicit occlusion filtering in the loss |
| Image resize and depth observations | Bilinear RGB resize to 640 x 192, ImageNet mean/std normalization and Euclidean LiDAR range cutoff of 80 m |
| Perturbations and seed | Independent uniform angle/translation draws, ZYX rotation composition, seed 42 and stable per-frame random generation; training perturbations resampled each epoch |
| Loss and shared-weight refinement | Average two training-step losses and use three inference steps; differentiate through continuous pose updates while rasterization/visibility is discrete and detached |
| Optimization and selection | Adam, 50 epochs, batch size 4, initial learning rate 1e-4, halved every 10 epochs; best checkpoint selected by the validation MAEs normalized by the two success thresholds |
| Constant-gate control | Initialize b at zero and learn it, freeze unused a, retain the valid-count indicator, and train independently with the same remaining hyperparameters |

Version 0.1.1 updates documentation and package metadata only. It does not introduce a new network, loss, or experimental configuration. Exact reproduction of the reported quantitative results requires the original data protocol and trained checkpoints or retraining under the original protocol. Software functionality checks are documented separately in [VALIDATION.md](VALIDATION.md).

## Limitations

* A high consistency score can arise from mutually wrong assignments; the gate is not calibrated correctness confidence.
* Discrete validity changes are not differentiable. It does not guarantee convergence over unrestricted initial offsets.
* This is residual calibration, not target-free recovery from arbitrary initial pose.
* The private test is controlled perturbation recovery relative to an independent reference, not direct measurement of natural mounting drift.
* The baseline adapter standardizes evaluation, but original third-party training code is outside the scope of this distribution.
