# Data preparation and availability

## Data availability

The KITTI Raw dataset is publicly available from its [official website](https://www.cvlibs.net/datasets/kitti/raw_data.php). The privately collected VLP-16–FLIR dataset and associated calibration information are not publicly distributed but are available from the corresponding author upon reasonable request.

Corresponding author: Tao Cao (caotao@sqxy.edu.cn).

This software package does not redistribute raw sensor recordings, original experiment manifests, trained checkpoints, or prediction outputs.

## Public KITTI Raw

Expected dataset layout (relative to data root):

```text
2011_09_26/
  calib_cam_to_cam.txt
  calib_velo_to_cam.txt
  2011_09_26_drive_0001_sync/
    image_02/data/0000000000.png
    velodyne_points/data/0000000000.bin
```

The camera is rectified image_02. The projection matrix P_rect_02 is factorized into K and the rectified camera baseline; the reference transform includes rectification R_rect_00 and the LiDAR-to-camera transform. RGB is resized to 640x192 and K is scaled accordingly. Point files contain float32 XYZI (KITTI binary) or float arrays Nx3/Nx4 in .npy format; only XYZ is consumed.

Train, validation and test drives must be disjoint. A drive list file optionally names the original drives:

```json
{"train": ["drive identifier from the actual split"],
 "val": ["different drive identifier"],
 "test": ["different drive identifier"]}
```

This example illustrates the schema only. The preparation utility saves sampled frame IDs and validation/test E matrices. For exact reproduction, use the original experiment manifests and fixed perturbations; contact the corresponding author for these protocol details. A newly generated split is intended for independent experiments unless its frame assignments and perturbations are confirmed to match. Use the same manifests for all compared methods.

## JSONL format for user-provided data

One JSON object per line; relative paths interpreted under --root. The image must already be undistorted or rectified, and K must correspond to that image before resizing. A synthetic schema example:

```json
{"id":"example/0001","drive":"example","image":"rgb/0001.png","points":"lidar/0001.npy","K":[[500,0,320],[0,500,240],[0,0,1]],"T_ref":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
```

The intrinsic and extrinsic numbers above are illustrative only. Replace with the actual sensor calibration. Optional E specifies a fixed perturbation transform; optional T0 specifies an actual initial extrinsic and bypasses synthetic perturbation. T0 without T_ref is supported for deployment inference. Do not mix labeled and unlabeled samples in one batch. For controlled three-trial testing with independently generated perturbations, omit E and T0, provide T_ref, and use --trials 3.

For the manuscript's VLP-16–FLIR protocol, use the 60 synchronized RGB–LiDAR pairs, camera intrinsics and independent reference extrinsics obtained through offline multi-pose checkerboard calibration. Apply three perturbations per pair (180 trials) using the original evaluation protocol. The independent reference calibration is an input to the evaluation, rather than a solver distributed with this package. Keep the real-platform results separate from KITTI metrics; repeated perturbations are trials, not independent sensor acquisitions. Private records and calibration metadata must not be redistributed without authorization.

## External comparator interface

Export protocol records include original image/cloud locations, K_original, K_resized, target image dimensions, range cutoff and T0. The test reference pose is deliberately omitted from this exported baseline input. External predicted transforms must use the same coordinate convention and sample ID. Example:

```json
{"id":"example/0001","T0":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],"T_pred":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
```

This is a schema example, not a baseline result. Score with the shared score-external command.
