# Wire Inspector — pretrained segmentation prototype

Upload a wire photo to a small Streamlit interface and inspect candidate burnmarks,
cuts, cracks and scratches. Download an overlay, class-index mask and JSON report.
No API key or custom training is needed to try the prototype.

## Google Colab notebook

Upload `Wire_Inspector_Colab.ipynb` to [Google Colab](https://colab.research.google.com/)
and run cells 1–6 in order. The notebook includes the inspection engine, so no
other project files are required. It installs dependencies, loads CLIPSeg, prompts
for one image, displays adjustable candidate masks, and downloads a results ZIP.
GPU is optional; CPU also works. Photos are processed in the Colab runtime.
Rerun cells 4–6 for another image, or only cells 5–6 to change mask settings.

## Windows quick start (Python 3.11 or 3.12 recommended)

Extract this folder, open a terminal inside it, then run:

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

Open http://localhost:8501 if your browser does not open automatically.
The explicit CPU installation avoids downloading unnecessary GPU dependencies.

## Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## What model is used?

One pretrained `CIDAS/clipseg-rd64-refined` model, using the Hugging Face Transformers
CLIPSeg implementation. It segments arbitrary text prompts. It is smaller than a
large multimodal language model, but it is **not a tiny mobile model**: expect a
several-hundred-MB model download plus the Python/PyTorch installation. Allow several
GB of free disk and RAM; exact use and inference time depend on the machine.
CPU works; CUDA is selected automatically when available. Six prompts run sequentially
to reduce peak memory. First run downloads weights; subsequent runs use the local cache.
Uploaded images are processed locally and are not sent to an inference API.
Model weights are not bundled in this ZIP.

## How to use

1. Upload a sharp JPG, PNG or WebP close-up, preferably one wire on a plain background.
2. Click **Analyze wire**. First run may take several minutes to download the model.
3. Inspect coloured candidate regions and descriptions.
4. Adjust thresholds to explore masks without rerunning the model.
5. Download results. Mask values: 0 unassigned; 1 burnmark; 2 cut; 3 crack; 4 scratch.

Images are EXIF-corrected and resized to at most 1600 pixels on the longest side.
All mask coordinates and area percentages refer to this processed image.
The underlying CLIPSeg map is lower resolution; upsampling does not restore tiny detail.
Descriptions are fixed candidate descriptions, not an LLM's verified observations.

## Interpreting results honestly

This model was not trained specifically to classify these wire defects. It may miss
small defects, confuse cracks with cuts, or highlight the entire wire. It cannot be
promised to “work well” without representative testing. Mask activation is not a
calibrated defect probability. Threshold defaults are exploratory, not validated.

The app gates defect masks to a text-prompted wire region, removes tiny connected
components, and assigns overlapping pixels to the strongest defect activation.
This is a heuristic, not proof that the chosen label is correct. Multiple labels
may appear in one image. A “Normal candidate” requires intact-wire activation over
at least half the localized wire and no surviving defect masks. Otherwise the app
returns uncertain or unable to inspect. Normal candidate is never a quality pass.
Background pixels and rejected candidate regions both have mask value zero.

For reliable deployment, collect real samples and pixel-mask annotations for all
four defect types plus normal wires. Split by physical wire/batch, then evaluate
per-class recall, false alarms and mask IoU on held-out data. Fine-tune a small
supervised segmentation model if this zero-shot baseline is insufficient.

## Troubleshooting

- Download failure: initial use requires access to huggingface.co and its model
  download endpoints. Retry on an unrestricted connection.
- Memory issue: close other apps or try a machine with more RAM.
- `torchvision::nms` errors: use the clean virtual environment above; this project
  does not need torchvision. In that environment, uninstall a mismatched torchvision.
- Empty masks: examine activation maps and lower thresholds experimentally.
  Never interpret an empty mask as proof of normality.
- Offline use: first download the model online, then run with the existing cache.

## Validation

`python -m unittest discover -s tests` tests mask gating, overlap, normal/uncertain
handling and exports using synthetic activations. These tests do not validate model
accuracy. No real wire dataset is supplied. See TEST_STATUS.md for build validation.

## References

- https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/clipseg
- https://huggingface.co/CIDAS/clipseg-rd64-refined
- https://github.com/timojl/clipseg

Review upstream model and dependency license terms before redistribution/commercial use.
