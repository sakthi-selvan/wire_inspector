"""One pretrained model; no external inference API or fabricated detections."""
import io
import numpy as np
from PIL import Image, ImageOps

MODEL_ID = 'CIDAS/clipseg-rd64-refined'
LABELS = ['Burnmark', 'Cut', 'Crack', 'Scratch']
PROMPTS = [
    'a burnt scorched patch on electrical wire insulation',
    'a cut opening in electrical wire insulation',
    'a crack or split in electrical wire insulation',
    'a scratch abrasion on electrical wire insulation',
    'an electrical wire or cable',
    'normal intact undamaged electrical wire insulation',
]
COLORS = [(240, 85, 60), (255, 185, 40), (170, 100, 255), (40, 190, 230)]
DESCRIPTIONS = [
    'Possible heat-damaged or scorched patch on the wire surface.',
    'Possible cut or opening in the wire surface.',
    'Possible narrow crack or split in the wire surface.',
    'Possible scratch or abrasion on the wire surface.',
]

def read_image(data):
    image = Image.open(io.BytesIO(data))
    if image.width * image.height > 20_000_000:
        raise ValueError('Please upload an image smaller than 20 megapixels.')
    image = ImageOps.exif_transpose(image).convert('RGB')
    image.thumbnail((1600, 1600))
    return image

class Inspector:
    def __init__(self):
        import torch
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = CLIPSegProcessor.from_pretrained(MODEL_ID)
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            MODEL_ID
        ).to(self.device).eval()

    def predict(self, image):
        import torch
        # Sequential prompts keep peak CPU/GPU memory lower than a six-image batch.
        maps = []
        with torch.inference_mode():
            for prompt in PROMPTS:
                inputs = self.processor(text=[prompt], images=[image],
                                        return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs, interpolate_pos_encoding=True).logits
                logits = logits.reshape(1, 1, *logits.shape[-2:])
                resized = torch.nn.functional.interpolate(
                    logits, size=(image.height, image.width),
                    mode='bilinear', align_corners=False)
                maps.append(resized.sigmoid()[0, 0].cpu().numpy())
        return np.stack(maps)

def summarize(image, maps, threshold=0.5, wire_threshold=0.35, min_pixels=30):
    """Heuristic candidate masks, never calibrated probabilities or quality decisions."""
    if maps.shape != (6, image.height, image.width):
        raise ValueError('Prediction dimensions do not match the image.')
    wire = maps[4] >= wire_threshold
    mask_ids = np.zeros(wire.shape, dtype=np.uint8)
    rows = []
    best = maps[:4].argmax(axis=0)
    for i, label in enumerate(LABELS):
        mask = wire & (maps[i] >= threshold) & (best == i)
        # Remove isolated pixel noise without smoothing away thin cracks.
        import cv2
        n, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
        keep = np.zeros(mask.shape, dtype=bool)
        for component in range(1, n):
            if stats[component, cv2.CC_STAT_AREA] >= min_pixels:
                keep |= components == component
        if keep.any():
            mask_ids[keep] = i + 1
            ys, xs = np.where(keep)
            rows.append({'defect': label, 'description': DESCRIPTIONS[i],
                         'mean_activation': round(float(maps[i][keep].mean()), 3),
                         'image_area_percent': round(100 * float(keep.mean()), 2),
                         'bounding_box_xyxy': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
    if wire.sum() < max(min_pixels, wire.size * 0.001):
        status = 'Unable to inspect — wire not reliably localized'
        rows, mask_ids = [], np.zeros_like(mask_ids)
    elif rows:
        status = 'Possible defect(s) — review highlighted regions'
    elif np.mean(maps[5][wire] >= threshold) >= 0.5:
        status = 'Normal candidate — no visible defect detected at these settings'
    else:
        status = 'Uncertain — no defect region passed the selected thresholds'
    overlay = np.array(image).copy()
    for i, color in enumerate(COLORS):
        region = mask_ids == i + 1
        overlay[region] = (overlay[region] * 0.5 + np.array(color) * 0.5).astype('uint8')
    return status, rows, Image.fromarray(overlay), Image.fromarray(mask_ids)


def png_bytes(image):
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()
