# Manuscript-to-code mapping

| Manuscript concept | Implementation |
|---|---|
| Eq. (1), T0 = E T*, delta T* = T* inv(T0) | geometry.perturbation; data.PairDataset |
| Eq. (2), project, retain nearest positive depth, validity mask | geometry.project; geometry.rasterize |
| Fig. 1/2 feature extraction | model.ResNet18; MultiScaleRGB; RCB; self-attention |
| Eq. (3), image-to-depth and depth-to-image associations | matching.directional_probabilities |
| Eq. (4), W = A * B transposed, chance-centered kappa | matching.reciprocal |
| Mutual residual feature updates | matching.ReciprocalMatcher |
| Eq. (5), bounded twist, monotonic gate and left SE(3) update | matching.ConfidenceGate; geometry.se3_exp; model.MultiAttCalib.step |
| Eq. (6), fixed reference-visible set and projection loss | losses.reference_points; losses.projection_loss |
| Two-step shared-weight training / three-step inference | engine.train; engine.evaluate |
| Table I, Fig. 5 and joint success | metrics; visualize.ablation_plot |
| Fig. 4/6 qualitative projection | visualize.qualitative; visualize.overlay |
| Real-world zero-shot protocol | PairDataset with three trials; evaluate |
| Paper revision, constant gate | matching.ConfidenceGate with constant_gate |

## Mathematical conventions

T transforms LiDAR coordinates into rectified camera coordinates using column vectors. Internally, batched points are stored as row vectors for convenience and transformed by X R^T+t. Rotation and translation are in radians and metres unless a metric explicitly converts them. Twist order is (omega_x, omega_y, omega_z, v_x, v_y, v_z); SE(3) translation is J(omega)v, not a direct substitution of v.

For Eq. (3), depth occupancy is area-pooled and the first softmax uses log m as an additive logit correction (gamma=1 by default). Invalid depth keys/queries contribute exactly zero. Empty valid sets produce zero in both directions. For Eq. (4), n=min(N_I,N_V); kappa=0 for n<2; otherwise sum(W) is centered by 1 and normalized by n-1.

Kappa is a global reciprocal consistency score, not a calibrated probability of correctness. Uniform/query-independent association in either direction yields sum(W)=1. Invalid depth updates are masked after the residual addition. The full gate computes sigmoid(softplus(a)*kappa+b), with the n>=2 indicator. The separate constant-gate control keeps that indicator and learns b, while removing kappa dependence. The no-gate control sets alpha=1 even when n<2, as defined by the manuscript.

The geometric loss uses points whose reference projection has positive depth and lies in the camera image. This set does not depend on predicted poses. Positive predicted depths are evaluated even if the predicted pixel is outside the image. Depth z<=epsilon uses the specified invalid penalty and slope. Smooth L1 is applied to the scalar Euclidean reprojection error with beta=3 pixels. An empty reference set contributes exactly zero.

## Metrics

For each trial, rotation MAE is the mean absolute ZYX Euler components (degrees) of R_pred R_ref^T. Translation MAE is the mean absolute components of t_pred-t_ref (centimetres). Dataset MAEs average those per-trial values; these are not geodesic rotation error or Euclidean translation norm. Joint success requires both rotation MAE <0.5 degrees and translation MAE <3 cm for the same trial.

The arms of the Fig. 5 box plots span the 5th–95th percentiles, boxes the interquartile range, means are diamonds and dashed red curves are separate marginal successes. Joint success is reported in summary.json, not the product of marginal success probabilities.
