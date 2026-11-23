import numpy as np
import pytest
from multiattcalib.data import kitti_calibration,make_splits,read_manifest


def test_kitti_includes_baseline(tmp_path):
    (tmp_path/"calib_cam_to_cam.txt").write_text(
        "P_rect_02: 10 0 5 2 0 10 4 0 0 0 1 0\nR_rect_00: 1 0 0 0 1 0 0 0 1\n")
    (tmp_path/"calib_velo_to_cam.txt").write_text("R: 1 0 0 0 1 0 0 0 1\nT: 1 2 3\n")
    K,T=kitti_calibration(tmp_path)
    np.testing.assert_allclose(T[:3,3],[1.2,2,3])


def test_split_disjoint_and_fixed(tmp_path):
    drives={str(i):[{"id":f"{i}/{j}","drive":str(i)} for j in range(3)] for i in range(6)}
    make_splits(drives,tmp_path,(6,3,3))
    train,val,test=[read_manifest(tmp_path/f"{k}.jsonl") for k in ("train","val","test")]
    assert {r["drive"] for r in train}.isdisjoint({r["drive"] for r in test+val})
    assert "E" not in train[0] and "E" in val[0] and "E" in test[0]
    with pytest.raises(ValueError,match="leakage"):
        make_splits(drives,tmp_path,(2,2,2),drive_lists={"train":["0"],"val":["0"],"test":["1"]})
