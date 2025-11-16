import types
import torch
import pandas as pd

import conformal_clip.benchmark as bench


class DummyVisionOnly:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        # Map zeros -> [1,0], ones -> [0,1]
        if x.ndim == 3:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        out = []
        for i in range(batch):
            s = float(x[i].sum().item())
            v = torch.tensor([1.0, 0.0], device=self.device) if s == 0.0 else torch.tensor([0.0, 1.0], device=self.device)
            out.append(v)
        return torch.stack(out, dim=0)


def _dummy_load_backend(backend, backend_model_id, device):
    model = DummyVisionOnly(device=device)

    def preprocess_fn(pil_img):  # unused because tests pass tensors
        return torch.zeros(3, 2, 2)

    return model, preprocess_fn, None


def test_benchmark_smoke_monkeypatch(monkeypatch):
    # Patch the loader used inside benchmark module
    monkeypatch.setattr(bench, "load_backend", _dummy_load_backend)

    device = torch.device("cpu")

    # Build tiny synthetic tensors: zeros = nominal, ones = defective
    train_nominal_images = [torch.zeros(3, 2, 2) for _ in range(2)]
    train_defective_images = [torch.ones(3, 2, 2) for _ in range(2)]
    calib_images = [torch.zeros(3, 2, 2), torch.ones(3, 2, 2)]
    calib_labels = ["Nominal", "Defective"]
    test_images = [torch.zeros(3, 2, 2), torch.ones(3, 2, 2)]
    test_labels = ["Nominal", "Defective"]

    cls_df, cp_df, cls_style, cp_style = bench.benchmark_models(
        train_nominal_images=train_nominal_images,
        train_defective_images=train_defective_images,
        calib_images=calib_images,
        calib_labels=calib_labels,
        test_images=test_images,
        test_labels=test_labels,
        device=device,
        seed=2025,
        backends=["dummy"],  # arbitrary string; only the loader is patched
        calibration_methods=(None,),
        conformal_modes=(None, "global"),
        alpha_list=(0.1,),
    )

    assert isinstance(cls_df, pd.DataFrame) and not cls_df.empty
    assert isinstance(cp_df, pd.DataFrame)
    # Styled outputs
    assert hasattr(cls_style, "to_html") and hasattr(cp_style, "to_html")

