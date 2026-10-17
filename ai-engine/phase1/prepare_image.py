"""Remove the background from a portrait with BiRefNet (MIT) and save it as RGBA.

Runs as its own process so its video memory is freed before TRELLIS.2 starts.
"""

import argparse

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

BIREFNET_REPO = "ZhengPeng7/BiRefNet"


def remove_background(image: Image.Image) -> Image.Image:
    model = AutoModelForImageSegmentation.from_pretrained(BIREFNET_REPO, trust_remote_code=True)
    model.to("cuda").eval().half()

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    rgb = image.convert("RGB")
    batch = transform(rgb).unsqueeze(0).to("cuda").half()
    with torch.no_grad():
        prediction = model(batch)[-1].sigmoid().float().cpu()

    mask = transforms.ToPILImage()(prediction[0].squeeze()).resize(rgb.size)
    rgba = rgb.copy()
    rgba.putalpha(mask)
    return rgba


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("output", help="RGBA PNG to write")
    args = parser.parse_args()

    remove_background(Image.open(args.image)).save(args.output)
    print(f"Background removed: {args.output}")


if __name__ == "__main__":
    main()
