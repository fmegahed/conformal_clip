import torch
import pandas as pd

from conformal_clip.zero_shot import evaluate_zero_shot_predictions


class DummyClipLike:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    @torch.no_grad()
    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Expecting 2 labels -> 2x2 identity
        n = token_ids.shape[0]
        assert n == 2
        emb = torch.eye(2, dtype=torch.float32, device=self.device)
        return emb

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        # If sum==0 -> class 0 vector; else class 1 vector
        if x.ndim == 3:  # [3,H,W]
            x = x.unsqueeze(0)
        batch = x.shape[0]
        out = []
        for i in range(batch):
            s = float(x[i].sum().item())
            v = torch.tensor([1.0, 0.0], device=self.device) if s == 0.0 else torch.tensor([0.0, 1.0], device=self.device)
            out.append(v)
        return torch.stack(out, dim=0)


def test_zero_shot_with_tokenize_fn_mock():
    device = torch.device("cpu")
    model = DummyClipLike(device=device)

    # Tokenize_fn returns a tensor of right batch size; values unused
    def tokenize_fn(labels):
        return torch.zeros((len(labels), 4), dtype=torch.long, device=device)

    # Two images: one zeros (class 0), one ones (class 1)
    im0 = torch.zeros(3, 2, 2)  # treated as preprocessed
    im1 = torch.ones(3, 2, 2)
    images = [im0, im1]
    labels = ["Nominal", "Defective"]
    label_counts = [1, 1]
    filenames = ["a.png", "b.png"]

    metrics_df, results_df = evaluate_zero_shot_predictions(
        labels=labels,
        label_counts=label_counts,
        test_images=images,
        test_image_filenames=filenames,
        model=model,
        device=device,
        tokenize_fn=tokenize_fn,
        save_confusion_matrix=False,
    )

    assert isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty
    assert isinstance(results_df, pd.DataFrame) and results_df.shape[0] == 2
    assert set(["true_label", "predicted_label"]).issubset(set(results_df.columns))

