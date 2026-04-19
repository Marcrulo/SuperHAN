"""
mediapipe_eval.py

Evaluate MediaPipe Hands as a baseline for landmark localisation, adding
one new condition to the existing four-condition comparison:

    5. MediaPipe on HR — MediaPipe applied to GT 128x128 image (best-case MP)
                         directly comparable to our Super-FAN which only sees
                         the degraded 32x32 LR input.

MediaPipe is a production-grade hand landmark detector, so comparing against
it is a meaningful real-world baseline — unlike the FAN-on-LR/HR conditions
which use our own architecture.

Joint ordering
──────────────
HaGRID landmarks are annotated in MediaPipe ordering, so no remapping is
needed between the dataset's ground truth and MediaPipe's predictions.

MediaPipe / HaGRID ordering (21 joints):
    0  = Wrist
    1-4  = Thumb  (CMC, MCP, IP,  TIP)
    5-8  = Index  (MCP, PIP, DIP, TIP)
    9-12 = Middle (MCP, PIP, DIP, TIP)
    13-16= Ring   (MCP, PIP, DIP, TIP)
    17-20= Pinky  (MCP, PIP, DIP, TIP)

MP_TO_HAGRID is therefore an identity mapping.
"""

import pathlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("WARNING: mediapipe not installed. Run: pip install mediapipe")

try:
    import tflite_runtime.interpreter as tflite
    _Interpreter = tflite.Interpreter
except ImportError:
    try:
        import tensorflow as tf
        _Interpreter = tf.lite.Interpreter
    except ImportError:
        _Interpreter = None


# ── Joint ordering ────────────────────────────────────────────────────────────

# HaGRID landmarks are annotated in MediaPipe ordering, so this is an
# identity mapping — no reordering needed between GT and MP predictions.
MP_TO_HAGRID = list(range(21))

# Keep old name as alias so any external code that imported MP_TO_FREIHAND
# continues to work without changes.
MP_TO_FREIHAND = MP_TO_HAGRID

# Build the inverse lookup for convenience
FREIHAND_TO_MP = list(range(21))


# ── MediaPipe runner ───────────────────────────────────────────────────────────

class MediaPipePredictor:
    """
    Runs the MediaPipe hand landmark TFLite model DIRECTLY on hand crops,
    bypassing the palm detection stage entirely.

    FreiHAND images are tight hand crops where the hand fills the whole frame,
    which breaks the palm detector. Loading the landmark model directly solves
    this — the landmark model expects a pre-cropped hand image (224x224) and
    returns 21 (x, y, z) normalised coordinates immediately.

    The model file is found automatically from the mediapipe package directory.
    No internet download required.
    """

    INPUT_SIZE = 224

    def __init__(self):
        if not MP_AVAILABLE:
            raise RuntimeError("mediapipe is not installed")
        if _Interpreter is None:
            raise RuntimeError(
                "TFLite runtime not found. Install one of:\n"
                "  pip install tflite-runtime\n"
                "  pip install tensorflow"
            )

        # Search for the landmark model — check common locations including
        # the project root, /tmp, and inside the mediapipe package itself.
        import os
        mp_path = pathlib.Path(mp.__file__).parent
        # Project root = two levels up from this file (eval/mediapipe_eval.py)
        project_root = pathlib.Path(__file__).parent.parent
        tmp = pathlib.Path("/tmp")
        candidates = [
            project_root / "hand_landmarker.task",
            tmp          / "hand_landmarker.task",
            mp_path / "modules/hand_landmark/hand_landmark_full.tflite",
            mp_path / "modules/hand_landmark/hand_landmark_lite.tflite",
            mp_path / "modules/hand_landmark/hand_landmark.tflite",
        ]
        model_path = next((str(p) for p in candidates if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError(
                "Could not find hand landmark model. Searched:\n"
                + "\n".join(f"  {c}" for c in candidates)
            )
        tqdm.write(f"  Using landmark model: {model_path}")
        self._model_path = model_path
        self._is_task    = model_path.endswith(".task")

        if self._is_task:
            # .task bundle — use MediaPipe Tasks API with IMAGE running mode
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            base_opts = mp_tasks.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.HandLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.1,
                min_hand_presence_confidence=0.1,
            )
            self._detector = mp_vision.HandLandmarker.create_from_options(opts)
        else:
            # Raw .tflite — load with TFLite interpreter directly
            if _Interpreter is None:
                raise RuntimeError(
                    "TFLite runtime not found. Install tflite-runtime or tensorflow."
                )
            self._interp = _Interpreter(model_path=model_path)
            self._interp.allocate_tensors()
            self._in_idx  = self._interp.get_input_details()[0]['index']
            self._out_idx = self._interp.get_output_details()[0]['index']

    def _preprocess(self, image_np: np.ndarray) -> np.ndarray:
        """Resize to INPUT_SIZE, normalise to [0,1], add batch dim."""
        import cv2
        resized = cv2.resize(image_np, (self.INPUT_SIZE, self.INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

    def predict(self, image_np: np.ndarray) -> Optional[np.ndarray]:
        """
        Run the landmark model on a single hand crop.

        Args:
            image_np: (H, W, 3) uint8 RGB image — tight hand crop.

        Returns:
            uv: (21, 2) float32 pixel coords in FreiHAND ordering,
                or None if .task bundle failed to detect (rare at conf=0.1).
        """
        H, W = image_np.shape[:2]
        image_np = np.ascontiguousarray(image_np, dtype=np.uint8)

        if self._is_task:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            result   = self._detector.detect(mp_image)
            if not result.hand_landmarks:
                return None
            landmarks = result.hand_landmarks[0]
            # Normalised [0,1] coords → pixel space
            uv = np.array([[lm.x * W, lm.y * H] for lm in landmarks],
                          dtype=np.float32)
        else:
            inp = self._preprocess(image_np)
            self._interp.set_tensor(self._in_idx, inp)
            self._interp.invoke()
            # Output (1, 63): 21 joints × 3 in [0, INPUT_SIZE] space
            raw = self._interp.get_tensor(self._out_idx).reshape(21, 3)
            uv  = np.zeros((21, 2), dtype=np.float32)
            uv[:, 0] = raw[:, 0] / self.INPUT_SIZE * W
            uv[:, 1] = raw[:, 1] / self.INPUT_SIZE * H

        # Both paths: MP and FreiHAND share identical joint ordering
        return uv[MP_TO_FREIHAND]

    def predict_batch(self, images_np: np.ndarray) -> tuple:
        """
        Run landmark model on a batch of images.

        Returns:
            uvs:      (B, 21, 2) predicted pixel coords in FreiHAND ordering.
            detected: (B,) boolean — False only if .task bundle finds no hand.
        """
        B        = len(images_np)
        H, W     = images_np.shape[1:3]
        uvs      = np.full((B, 21, 2), [W / 2, H / 2], dtype=np.float32)
        detected = np.zeros(B, dtype=bool)
        for i, img in enumerate(images_np):
            result = self.predict(img)
            if result is not None:
                uvs[i]      = result
                detected[i] = True
        return uvs, detected

    def close(self):
        if self._is_task and hasattr(self, '_detector'):
            self._detector.close()


# ── Tensor → uint8 numpy helper ────────────────────────────────────────────────

def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float tensor in [-1,1] → (H, W, 3) contiguous uint8 numpy."""
    arr = (t.cpu().clamp(-1, 1) + 1) / 2   # [0, 1]
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return np.ascontiguousarray(arr)        # mp.Image requires contiguous memory


# ── Full MediaPipe evaluation loop ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_mediapipe(
    generator,
    val_loader:  DataLoader,
    device:      str   = "cuda",
    hm_size:     int   = 128,
) -> dict:
    """
    Evaluate MediaPipe Hands under two conditions and return PCK results.

    Three conditions:
        5. MediaPipe on LR  — bilinear-upsampled 32x32 input, same degraded
                              image our FAN-on-LR baseline receives.
        6. MediaPipe on SR  — super-resolved 128x128 output from our generator,
                              same image SR-then-FAN receives.
        7. MediaPipe on HR  — GT 128x128, best-case input for MediaPipe.

    Detection failure handling:
        Failed detections are filled with the image centre (always wrong) so
        no samples are silently excluded — failures penalise the PCK score.

    Args:
        generator:        SRGenerator — used to produce SR images for cond. 6.
        val_loader:       DataLoader.
        device:           'cuda' or 'cpu'.
        hm_size:          Upsample size for LR images (128).

    Returns:
        dict with keys: 'mp_on_lr', 'mp_on_sr', 'mp_on_hr'
        Each sub-dict: pck_curve, auc, pck_at_02/05/10, detection_rate
    """
    from eval.metrics import pck_curve, auc, PCK_THRESHOLDS

    if not MP_AVAILABLE:
        raise RuntimeError("mediapipe is not installed. Run: pip install mediapipe")

    predictor = MediaPipePredictor()

    def _to_np(batch_tensors):
        return np.stack([_tensor_to_uint8(t) for t in batch_tensors])

    # Accumulators for all three conditions
    lr_pred, lr_gt, lr_vis, lr_det_list = [], [], [], []
    sr_pred, sr_gt, sr_vis, sr_det_list = [], [], [], []
    hr_pred, hr_gt, hr_vis, hr_det_list = [], [], [], []

    generator.eval()
    tqdm.write("  Running MediaPipe on LR, SR and HR images...")
    for batch in tqdm(val_loader, desc="  MediaPipe eval", unit="batch", leave=False):
        lr_img = batch['lr'].to(device)
        hr_img = batch['hr']
        gt_uv  = batch['uv'].numpy()
        vis    = batch['visible'].numpy()

        # Super-resolve LR → SR
        sr_img = generator(lr_img).cpu()

        # Upsample LR to hm_size for a fair comparison with FAN-on-LR
        lr_up  = F.interpolate(lr_img.cpu(), size=(hm_size, hm_size),
                               mode='bilinear', align_corners=False)

        pred_lr, det_lr = predictor.predict_batch(_to_np(lr_up))
        pred_sr, det_sr = predictor.predict_batch(_to_np(sr_img))
        pred_hr, det_hr = predictor.predict_batch(_to_np(hr_img))

        lr_pred.append(pred_lr); lr_gt.append(gt_uv)
        lr_vis.append(vis);      lr_det_list.append(det_lr)
        sr_pred.append(pred_sr); sr_gt.append(gt_uv)
        sr_vis.append(vis);      sr_det_list.append(det_sr)
        hr_pred.append(pred_hr); hr_gt.append(gt_uv)
        hr_vis.append(vis);      hr_det_list.append(det_hr)

    predictor.close()

    def _cat(lst): return np.concatenate(lst, axis=0)

    def _summarise(pred, gt, vis, det):
        pck_vals  = pck_curve(pred, gt, vis, PCK_THRESHOLDS)
        auc_score = auc(pck_vals)
        def pck_at(t):
            return float(pck_vals[np.argmin(np.abs(PCK_THRESHOLDS - t))])
        return {
            'pck_curve':      pck_vals,
            'auc':            auc_score,
            'pck_at_02':      pck_at(0.2),
            'pck_at_05':      pck_at(0.5),
            'pck_at_10':      pck_at(1.0),
            'detection_rate': float(det.mean()),
        }

    return {
        'mp_on_lr': _summarise(_cat(lr_pred), _cat(lr_gt), _cat(lr_vis), _cat(lr_det_list)),
        'mp_on_sr': _summarise(_cat(sr_pred), _cat(sr_gt), _cat(sr_vis), _cat(sr_det_list)),
        'mp_on_hr': _summarise(_cat(hr_pred), _cat(hr_gt), _cat(hr_vis), _cat(hr_det_list)),
    }


def print_mediapipe_results(mp_results: dict):
    """Print MediaPipe results in the same table format as print_results()."""
    conditions = [
        ('mp_on_lr', 'MediaPipe on LR'),
        ('mp_on_sr', 'MediaPipe on SR'),
        ('mp_on_hr', 'MediaPipe on HR (best-case)'),
    ]
    w = 46
    print("\n" + "=" * w)
    print("  MediaPipe Hands Baseline Results")
    print("=" * w)
    print(f"  {'Condition':<28} {'AUC':>5}  {'@0.2':>5}  {'@0.5':>5}  {'@1.0':>5}  {'Det%':>5}")
    print("-" * w)
    for key, label in conditions:
        r = mp_results[key]
        print(f"  {label:<28} "
              f"{r['auc']:.3f}  "
              f"{r['pck_at_02']:.3f}  "
              f"{r['pck_at_05']:.3f}  "
              f"{r['pck_at_10']:.3f}  "
              f"{r['detection_rate']*100:.1f}%")
    print("=" * w + "\n")