# ComfyUI-CorridorKey

ComfyUI custom node porting [CorridorKey](https://github.com/nikopueringer/CorridorKey) (GreenFormer)
green/blue-screen keying, following the pipeline design of the
[Luminia/CorridorKey](https://huggingface.co/spaces/Luminia/CorridorKey) Hugging Face Space: the node
generates its own alpha mask internally (fast classical HSV detection, AI via BiRefNet, or a hybrid
auto-fallback between the two) instead of requiring a pre-made mask input.

## Nodes

| Node | Description |
|---|---|
| CorridorKey (Luminia) Model Loader | Loads the GreenFormer checkpoint at a given inference resolution (1024/2048). |
| CorridorKey (Luminia) Keyer | Takes an `IMAGE` batch + the loaded model; generates its own mask, runs inference, despill, and despeckle. Outputs `composite`, `alpha`, `foreground`, `processed`. |

## Setup

1. `pip install -r requirements.txt` (inside your ComfyUI venv).
2. Download a GreenFormer checkpoint and place it in `ComfyUI/models/corridorkey/`:
   - Green screen: [nikopueringer/CorridorKey_v1.0](https://huggingface.co/nikopueringer/CorridorKey_v1.0) (`CorridorKey_v1.0.pth` or `.safetensors`)
   - Blue screen: [nikopueringer/CorridorKeyBlue_1.0](https://huggingface.co/nikopueringer/CorridorKeyBlue_1.0) (`CorridorKeyBlue_1.0.safetensors`)
3. Set the Keyer node's `screen_color` to match whichever checkpoint you loaded — it's a separate checkpoint
   per color, not a flag on the green one.

## Credits

- [CorridorKey](https://github.com/nikopueringer/CorridorKey) by Niko Pueringer / Corridor Digital — original
  GreenFormer model architecture and training.
- [Luminia/CorridorKey](https://huggingface.co/spaces/Luminia/CorridorKey) — pipeline design (mask-mode
  strategy, despill/despeckle math) this node's inference logic is ported from.
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — AI mask-generation model (`onnx-community/BiRefNet_lite-ONNX`).

## License

The vendored GreenFormer model architecture (`corridorkey_core/model_transformer.py`) originates from
CorridorKey, which is distributed under **CC-BY-NC-SA-4.0** (Attribution-NonCommercial-ShareAlike). This
repository's original glue code (nodes, mask/despill/despeckle utilities) is provided under the same terms
for consistency. Non-commercial use only; share adaptations under the same license; attribute the original
authors above.
