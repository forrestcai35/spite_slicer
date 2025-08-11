#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image


class SpriteSlicerError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice a sprite sheet image into individual sprites (grid or auto-detect).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Absolute path to the input sprite sheet image")
    parser.add_argument("--output", required=True, help="Absolute path to the output directory for sprites")

    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument("--auto", action="store_true", help="Enable auto-detection of non-uniform sprites")

    size_group = parser.add_argument_group("Sprite size (grid mode)")
    size_group.add_argument("--sprite-width", type=int, default=None, help="Width of each sprite in pixels")
    size_group.add_argument("--sprite-height", type=int, default=None, help="Height of each sprite in pixels")

    grid_group = parser.add_argument_group("Grid overrides (grid mode)")
    grid_group.add_argument("--cols", type=int, default=None, help="Number of columns to slice (auto if omitted)")
    grid_group.add_argument("--rows", type=int, default=None, help="Number of rows to slice (auto if omitted)")

    layout_group = parser.add_argument_group("Layout (grid mode)")
    layout_group.add_argument("--margin", type=int, default=0, help="Outer border in pixels around the sheet")
    layout_group.add_argument("--spacing", type=int, default=0, help="Gutter spacing in pixels between sprites")

    auto_group = parser.add_argument_group("Auto-detect options")
    auto_group.add_argument("--alpha-threshold", type=int, default=0, help="Foreground if alpha > this value (0-255)")
    auto_group.add_argument("--bg-color", type=str, default=None, help="Override background color as hex, e.g. #ffffff")
    auto_group.add_argument("--bg-tolerance", type=int, default=16, help="Tolerance for background color match (0-255)")
    auto_group.add_argument("--min-width", type=int, default=1, help="Minimum component width to keep")
    auto_group.add_argument("--min-height", type=int, default=1, help="Minimum component height to keep")
    auto_group.add_argument("--padding", type=int, default=0, help="Padding (pixels) around detected boxes")
    auto_group.add_argument("--sort", choices=["row", "x", "y", "area"], default="row", help="Sort order for output sprites")
    auto_group.add_argument("--connectivity", choices=[4, 8], type=int, default=4, help="Connectivity for component labeling")

    naming_group = parser.add_argument_group("Output naming")
    naming_group.add_argument("--prefix", default="sprite_", help="Filename prefix for output sprites")
    naming_group.add_argument("--zero-pad", type=int, default=None, help="Zero-padding digits for numbering (auto if omitted)")
    naming_group.add_argument("--extension", default="png", choices=["png", "jpg", "jpeg", "webp"], help="Output file extension")

    safety_group = parser.add_argument_group("Safety")
    safety_group.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing files")

    parser.add_argument("--verbose", action="store_true", help="Print more details during slicing")

    args = parser.parse_args()

    # Normalize and validate paths
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_path.is_file():
        raise SpriteSlicerError(f"Input image not found: {input_path}")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    args.input = str(input_path)
    args.output = str(output_dir)

    if not args.auto:
        if args.sprite_width is None or args.sprite_height is None:
            raise SpriteSlicerError("In grid mode, --sprite-width and --sprite-height are required. Use --auto for detection.")
        if args.sprite_width <= 0 or args.sprite_height <= 0:
            raise SpriteSlicerError("Sprite width and height must be positive integers")
        if args.cols is not None and args.cols <= 0:
            raise SpriteSlicerError("cols must be a positive integer if provided")
        if args.rows is not None and args.rows <= 0:
            raise SpriteSlicerError("rows must be a positive integer if provided")
        if args.margin < 0 or args.spacing < 0:
            raise SpriteSlicerError("margin and spacing cannot be negative")
    else:
        if args.alpha_threshold < 0 or args.alpha_threshold > 255:
            raise SpriteSlicerError("alpha-threshold must be in [0, 255]")
        if args.bg_tolerance < 0 or args.bg_tolerance > 255:
            raise SpriteSlicerError("bg-tolerance must be in [0, 255]")
        if args.min_width < 1 or args.min_height < 1:
            raise SpriteSlicerError("min-width and min-height must be >= 1")
        if args.padding < 0:
            raise SpriteSlicerError("padding cannot be negative")

    return args


def compute_grid(
    image_size: Tuple[int, int],
    sprite_width: int,
    sprite_height: int,
    margin: int,
    spacing: int,
    cols_override: Optional[int],
    rows_override: Optional[int],
) -> Tuple[int, int]:
    sheet_width, sheet_height = image_size

    usable_width = sheet_width - (margin * 2)
    usable_height = sheet_height - (margin * 2)
    if usable_width <= 0 or usable_height <= 0:
        raise SpriteSlicerError("Margin too large relative to image size")

    if cols_override is not None and rows_override is not None:
        return cols_override, rows_override

    def max_tiles(usable: int, tile: int, gap: int) -> int:
        if tile <= 0:
            return 0
        if usable < tile:
            return 0
        return 1 + max(0, (usable - tile) // (gap + tile if gap >= 0 else tile))

    auto_cols = max_tiles(usable_width, sprite_width, spacing)
    auto_rows = max_tiles(usable_height, sprite_height, spacing)

    if cols_override is not None:
        auto_cols = cols_override
    if rows_override is not None:
        auto_rows = rows_override

    if auto_cols <= 0 or auto_rows <= 0:
        raise SpriteSlicerError("Computed grid has zero columns or rows; check sizes, margin, and spacing")

    return auto_cols, auto_rows


# ---------- Auto-detect helpers ----------

def _parse_hex_color(text: str) -> Tuple[int, int, int]:
    t = text.strip().lstrip('#')
    if len(t) == 6:
        r = int(t[0:2], 16)
        g = int(t[2:4], 16)
        b = int(t[4:6], 16)
        return r, g, b
    raise SpriteSlicerError(f"Invalid --bg-color '{text}'. Use hex like #aabbcc")


def _infer_background_color_rgba(img_rgba: Image.Image, alpha_threshold: int) -> Tuple[Optional[Tuple[int, int, int]], bool]:
    arr = np.array(img_rgba)
    h, w, _ = arr.shape
    # Sample a 5-pixel border as flattened Nx4 arrays
    top = arr[0:5, :, :].reshape(-1, 4)
    bottom = arr[max(0, h-5):h, :, :].reshape(-1, 4)
    left = arr[:, 0:5, :].reshape(-1, 4)
    right = arr[:, max(0, w-5):w, :].reshape(-1, 4)
    border = np.concatenate([top, bottom, left, right], axis=0)
    alpha = border[:, 3]
    transparent_ratio = float((alpha <= alpha_threshold).sum()) / max(1, alpha.size)
    if transparent_ratio > 0.5:
        return None, True  # Use alpha-based foreground
    # Otherwise pick most common RGB from corners as background
    tl = arr[0:5, 0:5, :].reshape(-1, 4)
    tr = arr[0:5, max(0, w-5):w, :].reshape(-1, 4)
    bl = arr[max(0, h-5):h, 0:5, :].reshape(-1, 4)
    br = arr[max(0, h-5):h, max(0, w-5):w, :].reshape(-1, 4)
    corners = np.vstack([tl, tr, bl, br])
    rgb = corners[:, :3]
    # Quantize to reduce noise
    rgb_q = (rgb // 8) * 8
    uniq, counts = np.unique(rgb_q, axis=0, return_counts=True)
    bg_q = uniq[counts.argmax()]
    return (int(bg_q[0]), int(bg_q[1]), int(bg_q[2])), False


def _make_foreground_mask(img: Image.Image, alpha_threshold: int, bg_color: Optional[Tuple[int, int, int]], bg_tol: int) -> np.ndarray:
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    if bg_color is None:
        return alpha > alpha_threshold
    rgb = arr[:, :, :3].astype(np.int16)
    br, bgc, bb = bg_color
    dist = np.abs(rgb[:, :, 0] - br) + np.abs(rgb[:, :, 1] - bgc) + np.abs(rgb[:, :, 2] - bb)
    mask = dist > (bg_tol * 3)
    mask &= alpha > alpha_threshold
    return mask


def _find_components(mask: np.ndarray, connectivity: int) -> List[Tuple[int, int, int, int]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    boxes: List[Tuple[int, int, int, int]] = []

    if connectivity == 8:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            min_x = max_x = x
            min_y = max_y = y
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                if cx < min_x: min_x = cx
                if cx > max_x: max_x = cx
                if cy < min_y: min_y = cy
                if cy > max_y: max_y = cy
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            boxes.append((min_x, min_y, max_x + 1, max_y + 1))
    return boxes


def _pad_and_filter_boxes(boxes: List[Tuple[int, int, int, int]], w: int, h: int, padding: int, min_w: int, min_h: int) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for (x0, y0, x1, y1) in boxes:
        x0p = max(0, x0 - padding)
        y0p = max(0, y0 - padding)
        x1p = min(w, x1 + padding)
        y1p = min(h, y1 + padding)
        if (x1p - x0p) >= min_w and (y1p - y0p) >= min_h:
            out.append((x0p, y0p, x1p, y1p))
    return out


def _sort_boxes(boxes: List[Tuple[int, int, int, int]], how: str) -> List[Tuple[int, int, int, int]]:
    if how == "x":
        return sorted(boxes, key=lambda b: b[0])
    if how == "y":
        return sorted(boxes, key=lambda b: b[1])
    if how == "area":
        return sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def slice_sprites_auto(args: argparse.Namespace) -> int:
    img = Image.open(args.input).convert("RGBA")
    w, h = img.size

    # Decide background strategy
    bg_color = None
    use_alpha = True
    if args.bg_color:
        bg_color = _parse_hex_color(args.bg_color)
        use_alpha = False
    else:
        inferred_bg, should_use_alpha = _infer_background_color_rgba(img, args.alpha_threshold)
        use_alpha = should_use_alpha
        bg_color = None if use_alpha else inferred_bg

    mask = _make_foreground_mask(img, args.alpha_threshold, bg_color, args.bg_tolerance)

    boxes = _find_components(mask, args.connectivity)
    boxes = _pad_and_filter_boxes(boxes, w, h, args.padding, args.min_width, args.min_height)
    boxes = _sort_boxes(boxes, args.sort)

    total = len(boxes)
    zero_pad = args.zero_pad if args.zero_pad is not None else int(math.log10(total)) + 1 if total > 0 else 1

    if args.verbose:
        mode_desc = "alpha" if use_alpha else f"bg color {bg_color}±{args.bg_tolerance}"
        print(f"Detected {total} sprites using {mode_desc}. Output: {args.output}")

    for index, (x0, y0, x1, y1) in enumerate(boxes):
        crop = img.crop((x0, y0, x1, y1))
        filename = f"{args.prefix}{str(index).zfill(zero_pad)}.{args.extension}"
        out_path = Path(args.output) / filename
        if args.no_overwrite and out_path.exists():
            if args.verbose:
                print(f"Skip existing: {out_path}")
            continue
        save_params = {}
        if args.extension.lower() in {"jpg", "jpeg"}:
            if crop.mode in ("RGBA", "LA") or (crop.mode == "P" and "transparency" in crop.info):
                crop = crop.convert("RGB")
            save_params["quality"] = 95
            save_params["optimize"] = True
        elif args.extension.lower() == "png":
            save_params["optimize"] = True
        crop.save(str(out_path), **save_params)
        if args.verbose:
            print(f"Saved: {out_path} ({x0},{y0})-({x1},{y1})")

    if args.verbose:
        print(f"Done. Wrote {total} sprites to {args.output}")
    return 0


def slice_sprites_grid(args: argparse.Namespace) -> int:
    image = Image.open(args.input)
    sheet_width, sheet_height = image.size

    cols, rows = compute_grid(
        image_size=(sheet_width, sheet_height),
        sprite_width=args.sprite_width,
        sprite_height=args.sprite_height,
        margin=args.margin,
        spacing=args.spacing,
        cols_override=args.cols,
        rows_override=args.rows,
    )

    total = cols * rows
    zero_pad = args.zero_pad if args.zero_pad is not None else int(math.log10(total)) + 1 if total > 0 else 1

    if args.verbose:
        print(
            f"Input: {args.input}\nSize: {sheet_width}x{sheet_height}\nSprite: {args.sprite_width}x{args.sprite_height}\n"
            f"Margin: {args.margin}, Spacing: {args.spacing}\nGrid: {cols} cols x {rows} rows (total {total})\n"
            f"Output: {args.output}, Prefix: {args.prefix}, Extension: {args.extension}, Zero-pad: {zero_pad}"
        )

    index = 0
    for row in range(rows):
        for col in range(cols):
            left = args.margin + col * (args.sprite_width + args.spacing)
            top = args.margin + row * (args.sprite_height + args.spacing)
            right = left + args.sprite_width
            bottom = top + args.sprite_height
            if right > sheet_width - args.margin + args.spacing or bottom > sheet_height - args.margin + args.spacing:
                continue
            crop = image.crop((left, top, right, bottom))
            filename = f"{args.prefix}{str(index).zfill(zero_pad)}.{args.extension}"
            out_path = Path(args.output) / filename
            if args.no_overwrite and out_path.exists():
                if args.verbose:
                    print(f"Skip existing: {out_path}")
                index += 1
                continue
            save_params = {}
            if args.extension.lower() in {"jpg", "jpeg"}:
                if crop.mode in ("RGBA", "LA") or (crop.mode == "P" and "transparency" in crop.info):
                    crop = crop.convert("RGB")
                save_params["quality"] = 95
                save_params["optimize"] = True
            elif args.extension.lower() == "png":
                save_params["optimize"] = True
            crop.save(str(out_path), **save_params)
            if args.verbose:
                print(f"Saved: {out_path}")
            index += 1

    if args.verbose:
        print(f"Done. Wrote {index} sprites to {args.output}")
    return 0


def main() -> None:
    try:
        args = parse_args()
        exit_code = slice_sprites_auto(args) if args.auto else slice_sprites_grid(args)
    except SpriteSlicerError as e:
        print(f"Error: {e}")
        exit_code = 2
    except KeyboardInterrupt:
        print("Interrupted")
        exit_code = 130
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main() 