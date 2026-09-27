"""Generate reproducible image variations, not new independent defect examples."""
import argparse
import csv
import hashlib
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def augment(image, rng, size):
    settings = {
        'angle': rng.uniform(-15, 15),
        'horizontal_flip': rng.choice([False, True]),
        'vertical_flip': rng.choice([False, True]),
        'brightness': rng.uniform(0.8, 1.2),
        'contrast': rng.uniform(0.85, 1.15),
        'color': rng.uniform(0.9, 1.1),
        'blur_radius': rng.uniform(0, 0.5),
        'scale': rng.uniform(0.85, 1.0),
        'jpeg_quality': rng.randint(90, 96),
    }
    result = image.copy()
    if settings['horizontal_flip']:
        result = ImageOps.mirror(result)
    if settings['vertical_flip']:
        result = ImageOps.flip(result)
    # Expanding the canvas preserves the full photo rather than cropping defects.
    result = result.rotate(settings['angle'], Image.Resampling.BICUBIC,
                           expand=True, fillcolor=(128, 128, 128))
    result = ImageEnhance.Brightness(result).enhance(settings['brightness'])
    result = ImageEnhance.Contrast(result).enhance(settings['contrast'])
    result = ImageEnhance.Color(result).enhance(settings['color'])
    result = result.filter(ImageFilter.GaussianBlur(settings['blur_radius']))
    side = max(1, round(size * settings['scale']))
    result = ImageOps.contain(result, (side, side), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (size, size), (128, 128, 128))
    x = rng.randint(0, size - result.width)
    y = rng.randint(0, size - result.height)
    settings.update(offset_x=x, offset_y=y)
    canvas.paste(result, (x, y))
    output = io.BytesIO()
    canvas.save(output, format='JPEG', quality=settings['jpeg_quality'])
    return output.getvalue(), settings


def generate(input_dir, output_dir, count=10000, size=512, seed=42):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if count < 1 or size < 64:
        raise ValueError('Count must be positive and size must be at least 64.')
    if not input_dir.is_dir():
        raise ValueError(f'Input directory does not exist: {input_dir}')
    if output_dir == input_dir or output_dir in input_dir.parents:
        raise ValueError('Output must not be the input directory or its parent.')
    # Only immediate input files are read; previous output subfolders are excluded.
    sources = sorted(p for p in input_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
    if not sources:
        raise ValueError('No JPG, PNG or WebP images found in the input directory.')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Output directory is not empty. Choose a new --output folder.')
    images, source_info = [], []
    for path in sources:
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
        source_info.append({'filename': path.name,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'original_size': list(image.size)})
        image.thumbnail((size * 2, size * 2), Image.Resampling.LANCZOS)
        images.append(image)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng, seen = random.Random(seed), set()
    counts = {path.name: 0 for path in sources}
    with (output_dir / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'source', 'source_sha256',
                                              'sha256', 'seed', 'transforms'])
        writer.writeheader()
        for index in range(count):
            source_index = index % len(sources)
            for attempt in range(100):
                data, settings = augment(images[source_index], rng, size)
                digest = hashlib.sha256(data).hexdigest()
                if digest not in seen:
                    break
            else:
                raise RuntimeError('Could not generate a unique image after 100 attempts.')
            seen.add(digest)
            name = f'aug_{index + 1:05d}.jpg'
            (output_dir / name).write_bytes(data)
            source = source_info[source_index]
            writer.writerow({'filename': name, 'source': source['filename'],
                             'source_sha256': source['sha256'], 'sha256': digest,
                             'seed': seed, 'transforms': json.dumps(settings, sort_keys=True)})
            counts[source['filename']] += 1
            if (index + 1) % 500 == 0 or index + 1 == count:
                print(f'Generated {index + 1:,}/{count:,}', flush=True)
    summary = {'count': count, 'size': [size, size], 'seed': seed,
               'sources': source_info, 'counts_by_source': counts,
               'note': 'Augmented variations only. No new defect labels or segmentation masks. '
                       'Split by physical cable/source before augmentation, never randomly '
                       'across variants. Sources from the same cable belong in the same split.'}
    (output_dir / 'dataset_info.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'Saved images, manifest.csv and dataset_info.json to {output_dir}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('data/augmented'))
    parser.add_argument('--count', type=int, default=10000, help='Total output images, not per source')
    parser.add_argument('--size', type=int, default=512, help='Output width and height in pixels')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        generate(args.input, args.output, args.count, args.size, args.seed)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
