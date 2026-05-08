"""
학습 이미지 오프라인 증강 스크립트

원본 정상 이미지를 증강하여 학습 데이터 수를 늘림.
드론 촬영 환경을 고려한 증강: 조명 변화, 미세 회전, 블러, 노이즈 등

사용법:
    python augment_train.py --input data/train --output data/glass_format/spot/train/good --repeat 5
"""
import os
import argparse
import random
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms


def get_augment_transforms():
    """드론 촬영 환경에 맞는 증강 파이프라인 목록"""
    return [
        # 1) 조명 변화 (밝기 + 대비)
        transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.2, hue=0.02),
        # 2) 미세 회전 (드론 기울어짐)
        transforms.RandomRotation(degrees=8),
        # 3) 미세 이동 + 스케일 변화 (고도 변화)
        transforms.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    ]


def augment_image(image):
    """이미지 하나에 랜덤 증강 2~3개 조합 적용"""
    aug_list = get_augment_transforms()
    n_augs = random.randint(2, 3)
    selected = random.sample(aug_list, n_augs)
    transform = transforms.Compose(selected)
    return transform(image)


def main():
    parser = argparse.ArgumentParser(description="학습 이미지 오프라인 증강")
    parser.add_argument("--input", type=str, required=True, help="원본 이미지 폴더")
    parser.add_argument("--output", type=str, required=True, help="증강 이미지 저장 폴더")
    parser.add_argument("--repeat", type=int, default=5, help="원본 1장당 증강 이미지 수")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = sorted([f for f in os.listdir(args.input) if f.lower().endswith(valid_ext)])
    print(f"원본 이미지: {len(image_files)}장, repeat: {args.repeat}x")
    print(f"예상 총 이미지: {len(image_files) + len(image_files) * args.repeat}장 (원본 + 증강)")

    # 원본 이미지 복사
    for f in image_files:
        src = os.path.join(args.input, f)
        dst = os.path.join(args.output, f)
        if not os.path.exists(dst):
            Image.open(src).save(dst)

    # 증강 이미지 생성
    count = 0
    for f in image_files:
        img = Image.open(os.path.join(args.input, f)).convert("RGB")
        name, ext = os.path.splitext(f)
        for i in range(args.repeat):
            aug_img = augment_image(img)
            save_name = f"{name}_aug{i:02d}{ext}"
            aug_img.save(os.path.join(args.output, save_name))
            count += 1

        if (image_files.index(f) + 1) % 10 == 0:
            print(f"  {image_files.index(f) + 1}/{len(image_files)} 완료")

    print(f"\n완료! 증강 이미지 {count}장 생성")
    print(f"총 이미지: {len(image_files) + count}장 ({args.output})")


if __name__ == "__main__":
    main()
