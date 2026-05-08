"""
LabelMe JSON → 바이너리 마스크 변환

사용법:
    python convert_labelme_to_mask.py
    python convert_labelme_to_mask.py --input data/glass_format/spot/ground_truth/bad_json --output data/glass_format/spot/ground_truth/bad
"""
import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw


def json_to_mask(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    h = data['imageHeight']
    w = data['imageWidth']
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)

    POINT_RADIUS = 10
    LINE_WIDTH = 8

    for shape in data['shapes']:
        points = [tuple(p) for p in shape['points']]
        if shape['shape_type'] == 'polygon':
            draw.polygon(points, fill=255)
        elif shape['shape_type'] == 'circle':
            cx, cy = points[0]
            ex, ey = points[1]
            r = ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        elif shape['shape_type'] == 'rectangle':
            draw.rectangle([points[0], points[1]], fill=255)
        elif shape['shape_type'] == 'point':
            cx, cy = points[0]
            draw.ellipse([cx - POINT_RADIUS, cy - POINT_RADIUS,
                          cx + POINT_RADIUS, cy + POINT_RADIUS], fill=255)
        elif shape['shape_type'] == 'linestrip':
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=255, width=LINE_WIDTH)

    mask.save(output_path)
    white_ratio = np.mean(np.array(mask) > 0) * 100
    return white_ratio


def main():
    parser = argparse.ArgumentParser(description="LabelMe JSON → 바이너리 마스크")
    parser.add_argument("--input", type=str,
                        default=r"C:\Users\User\github\spot_catcher\data\glass_format\spot\ground_truth\bad_json",
                        help="LabelMe JSON 폴더")
    parser.add_argument("--output", type=str,
                        default=r"C:\Users\User\github\spot_catcher\data\glass_format\spot\ground_truth\bad",
                        help="마스크 저장 폴더")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    json_files = sorted([f for f in os.listdir(args.input) if f.endswith('.json')])
    print(f"총 {len(json_files)}개 JSON 변환 시작")

    for f in json_files:
        json_path = os.path.join(args.input, f)
        img_name = os.path.splitext(f)[0] + '.png'
        output_path = os.path.join(args.output, img_name)

        white_ratio = json_to_mask(json_path, output_path)
        print(f"  {img_name:30s} defect={white_ratio:.1f}%")

    print(f"\n완료! 마스크 저장: {args.output}")


if __name__ == "__main__":
    main()
