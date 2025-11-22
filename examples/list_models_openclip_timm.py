"""
Quick discovery script listing available OpenCLIP and timm model names.

Run:
  python examples/list_models_openclip_timm.py
"""

import sys


def list_openclip():
    import open_clip

    # Build {model_name: set(pretrained_tags)} in a version-agnostic way
    by_model = {}

    fn = getattr(open_clip, "list_pretrained_tags_by_model", None)
    if callable(fn):
        try:
            res = fn()  # type: ignore[call-arg]
        except TypeError:
            res = None
        else:
            if isinstance(res, dict):
                by_model = {k: set(v) for k, v in res.items()}

    if not by_model:
        all_tags = open_clip.list_pretrained()
        for item in all_tags:
            if isinstance(item, tuple) and len(item) >= 2:
                name, tag = item[0], item[1]
                by_model.setdefault(name, set()).add(tag)

    names = sorted(by_model.keys())
    print(f"[open_clip] Found {len(names)} built-in model names with pretrained tags.")
    for name in names[:30]:
        tags = sorted(by_model.get(name, []))
        tag_preview = ", ".join(tags[:3]) + (" ..." if len(tags) > 3 else "")
        print(f"  - {name}: {tag_preview}")


def list_timm():
    try:
        import timm
    except Exception as e:
        print("[timm] Not installed or failed to import:", e)
        return

    names = timm.list_models(pretrained=True)
    print(f"[timm] Found {len(names)} pretrained models.")
    print("  First 30:")
    for n in names[:30]:
        print("   ", n)

    # Family filters
    families = {
        "mobilenet*": timm.list_models("mobilenet*", pretrained=True)[:10],
        "convnext*": timm.list_models("convnext*", pretrained=True)[:10],
        "vit*": timm.list_models("vit*", pretrained=True)[:10],
    }
    for pat, lst in families.items():
        print(f"  {pat}: {lst}")


def main():
    print("=== OpenCLIP models ===")
    list_openclip()
    print("\n=== timm models ===")
    list_timm()


if __name__ == "__main__":
    sys.exit(main())
