"""2-minute check (run FIRST on the pod, before the full experiment):
confirms the current QUANT model loads AND the layer-N residual-stream hook
returns a real activation. If this passes for awq/gptq, the projection
measurement is valid; if it fails, we stop before spending on full runs.

Usage on pod:
    MODEL=8b QUANT=awq  python hook_check.py
    MODEL=8b QUANT=gptq python hook_check.py
"""
import os, torch
from refusal_direction import load_model, get_device, MODEL_ID

LAYER = int(os.environ.get("CHECK_LAYER", "14"))  # any mid layer for the test

_cap = {}
def hook(m, i, o):
    h = o[0] if isinstance(o, tuple) else o
    if h.shape[1] > 1:
        _cap["act"] = h[0, -1, :].detach().float()

device = get_device()
tok, model = load_model(device)
nlayers = len(model.model.layers)
L = min(LAYER, nlayers - 1)
print(f"Model loaded. Decoder layers: {nlayers}. Testing hook on layer {L}.")

handle = model.model.layers[L].register_forward_hook(hook)
text = tok.apply_chat_template(
    [{"role": "user", "content": "Explain how a bicycle works."}],
    add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt").to(device)
with torch.no_grad():
    model(**ids)
handle.remove()

if "act" in _cap:
    a = _cap["act"]
    print(f"HOOK OK: captured activation shape {tuple(a.shape)}, "
          f"norm {a.norm().item():.2f}, finite={torch.isfinite(a).all().item()}")
    print("=> projection measurement will work on this precision. Proceed.")
else:
    print("HOOK FAILED: no activation captured. Do NOT run full experiment; "
          "the quantized layers are not exposing the residual stream as expected.")
