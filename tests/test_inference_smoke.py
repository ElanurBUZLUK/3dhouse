import os
from pathlib import Path

import numpy as np
import cv2

from src.pipeline import run_pipeline


def _ensure_sample_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        img = np.full((256, 256, 3), 255, dtype=np.uint8)
        # simple house-like sketch: rectangle + triangle roof + door
        cv2.rectangle(img, (64, 96), (192, 208), (0, 0, 0), 2)
        pts = np.array([[64, 96], [128, 48], [192, 96]], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=2)
        cv2.rectangle(img, (118, 160), (138, 208), (0, 0, 0), 2)
        cv2.imwrite(str(path), img)


def test_pipeline_smoke(tmp_path):
    sample = Path('data/raw/sample.jpg')
    _ensure_sample_image(sample)

    out_dir = tmp_path / 'out'
    res = run_pipeline(str(sample), str(out_dir), export_format='gltf')

    assert res['success'] is True
    assert isinstance(res.get('processing_time'), float)

    # mask preview should exist if segmentation step succeeded
    mask_png = out_dir / 'mask.png'
    assert mask_png.exists(), 'mask.png not found'

    # export should produce at least one file (GLB through exporter)
    files = res.get('output_files', [])
    assert files and Path(files[0]).exists(), 'exported model not found'
