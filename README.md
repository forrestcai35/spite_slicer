# Sprite Slicer

A simple CLI to slice a sprite sheet into individual sprite images.

## Install

```bash
cd "/Users/forrest/Projects/Sprite slicer"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Grid mode (uniform tiles):

```bash
python slice_sprites.py \
  --input /absolute/path/to/sheet.png \
  --output /absolute/path/to/output_dir \
  --sprite-width 32 \
  --sprite-height 32
```

With margins and spacing (gutters between sprites):

```bash
python slice_sprites.py \
  --input /absolute/path/to/sheet.png \
  --output /absolute/path/to/output_dir \
  --sprite-width 64 \
  --sprite-height 64 \
  --margin 0 \
  --spacing 0
```

Specify explicit rows/cols (overrides auto grid inference):

```bash
python slice_sprites.py \
  --input /absolute/path/to/sheet.png \
  --output /absolute/path/to/output_dir \
  --sprite-width 32 \
  --sprite-height 32 \
  --cols 8 \
  --rows 4
```

Auto-detect non-uniform sprites (connected components):

```bash
python slice_sprites.py \
  --auto \
  --input /absolute/path/to/sheet.png \
  --output /absolute/path/to/output_dir \
  --alpha-threshold 0 \
  --min-width 8 --min-height 8 \
  --padding 1 --sort row --verbose
```

- If the image has transparency, pixels with alpha > `--alpha-threshold` are treated as foreground.
- If not, background color is inferred from image corners; override with `--bg-color` (hex like `#ffffff`) and `--bg-tolerance`.

Additional options:

- `--prefix`: filename prefix (default: `sprite_`)
- `--zero-pad`: digits for numbering (default auto)
- `--margin`: outer border around the sheet (default: 0) [grid mode]
- `--spacing`: gutter between sprites (default: 0) [grid mode]
- `--no-overwrite`: prevent overwriting existing files
- `--verbose`: print more details
