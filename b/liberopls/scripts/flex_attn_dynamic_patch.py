"""Force flex_attention torch.compile(dynamic=True) before GAM imports.

Import this module (or set PYTHONPATH to b/liberopls and PYTHONSTARTUP) so that
variable policy batch sizes do not trip inductor Batch-dimension asserts.
The official future_predictor compiles flex_attention with dynamic=False.
"""

from __future__ import annotations


def apply() -> None:
    import torch

    try:
        from torch.nn.attention.flex_attention import flex_attention as raw
    except Exception as exc:  # noqa: BLE001
        print(f"[liberopls.flex_patch] flex_attention unavailable: {exc}")
        return

    compiled = torch.compile(raw, dynamic=True)
    # Patch both modeling modules if already imported; otherwise seed globals
    # that future_predictor/da3_giant_encoder will pick up only if we inject
    # before their import. Prefer importing this file first via sitecustomize.
    import sys

    for mod_name in (
        "robot.modeling.future_predictor",
        "robot.modeling.da3_giant_encoder",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "_flex_attention"):
            mod._flex_attention = compiled
            print(f"[liberopls.flex_patch] patched {mod_name}._flex_attention dynamic=True")

    # Also stash for early importers
    sys.modules.setdefault("_liberopls_flex_compiled", type(sys)("x"))
    print("[liberopls.flex_patch] compiled flex_attention with dynamic=True")


if __name__ == "__main__":
    apply()
