"""
Example showing how to:
  1) List available OpenCLIP model names and pretrained tags.
  2) Load a custom OpenCLIP backbone via the `custom-clip` backend.

This script does not depend on the textile dataset and should run quickly
once open-clip-torch is installed.
"""

import torch

from conformal_clip import load_backend


def list_some_openclip_models(max_rows: int = 15) -> None:
    import open_clip

    by_model = open_clip.list_pretrained()
    print("Some available (model, tag) pairs from open_clip.list_pretrained():")
    for model_name, tag, *_ in by_model[:max_rows]:
        print(f"  - {model_name:30s}  |  {tag}")


def load_custom_clip_example() -> None:
    """
    Load a custom OpenCLIP model via the `custom-clip` backend.

    We pick a ViT-B/32 variant as an example; you can substitute any
    (model, tag) pair returned by open_clip.list_pretrained().
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example: ViT-B/32 with a specific pretrained tag.
    # Replace "ViT-B-32-quickgelu@openai" with any other "model@tag" string
    # that appears in open_clip.list_pretrained().
    backend = "custom-clip"
    backend_model_id = "ViT-B-32-quickgelu@openai"

    model, preprocess_fn, tokenize_fn = load_backend(
        backend=backend,
        backend_model_id=backend_model_id,
        device=device,
    )

    print(f"Loaded custom backend '{backend}' with id '{backend_model_id}' on {device}.")
    print("  - model type:", type(model))
    print("  - has tokenize_fn:", tokenize_fn is not None)
    print("You can now use this (model, preprocess_fn, tokenize_fn) triplet in the")
    print("few_shot_fault_classification_conformal or zero-shot utilities.")


def main() -> None:
    list_some_openclip_models()
    print("\n---\n")
    load_custom_clip_example()


if __name__ == "__main__":
    main()

