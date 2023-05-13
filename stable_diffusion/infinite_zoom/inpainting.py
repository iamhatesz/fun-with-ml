from typing import Callable

import torch
from PIL import Image
from torchvision import transforms

_to_pillow = transforms.ToPILImage()
_to_tensor = transforms.PILToTensor()

Generator = Callable[[Image.Image, Image.Image], Image.Image]
Animation = list[Image.Image]
OutpaintMethod = Callable[[Image.Image, int, Generator], Animation]


def outpaint_sequentially(
    initial_image: Image.Image,
    outpaint_size: int,
    generator: Generator,
) -> Image.Image:
    _to_pillow = transforms.ToPILImage()
    _to_tensor = transforms.PILToTensor()

    resize_transforms = [
        transforms.Pad([outpaint_size, 0, 0, 0], padding_mode="symmetric"),
        transforms.Pad([0, outpaint_size, 0, 0], padding_mode="symmetric"),
        transforms.Pad([0, 0, outpaint_size, 0], padding_mode="symmetric"),
        transforms.Pad([0, 0, 0, outpaint_size], padding_mode="symmetric"),
    ]

    def _left_mask(image: Image.Image) -> Image.Image:
        xs = torch.arange(image.width)
        mask = torch.zeros_like(_to_tensor(image))
        mask[:, :, xs < outpaint_size] = 255
        return _to_pillow(mask)

    def _top_mask(image: Image.Image) -> Image.Image:
        ys = torch.arange(image.height)
        mask = torch.zeros_like(_to_tensor(image))
        mask[:, ys < outpaint_size] = 255
        return _to_pillow(mask)

    def _right_mask(image: Image.Image) -> Image.Image:
        xs = torch.arange(image.width)
        mask = torch.zeros_like(_to_tensor(image))
        mask[:, :, xs > initial_image.width + outpaint_size] = 255
        return _to_pillow(mask)

    def _bottom_mask(image: Image.Image) -> Image.Image:
        ys = torch.arange(image.height)
        mask = torch.zeros_like(_to_tensor(image))
        mask[:, ys > initial_image.height + outpaint_size] = 255
        return _to_pillow(mask)

    mask_generators = [
        _left_mask,
        _top_mask,
        _right_mask,
        _bottom_mask,
    ]

    image = initial_image
    for resize, mask_gen in zip(resize_transforms, mask_generators):
        image_raw = _to_tensor(image)
        padded_image = _to_pillow(resize(image_raw))
        mask = mask_gen(padded_image)
        image = generator(padded_image, mask)

    return image


def outpaint_direct(
    initial_image: Image.Image,
    outpaint_size: int,
    generator: Generator,
) -> Image.Image:
    resize = transforms.Pad(outpaint_size, padding_mode="symmetric")
    image_raw = resize(_to_tensor(initial_image))
    padded_image = _to_pillow(image_raw)

    ixs = torch.arange(padded_image.width)
    mask = torch.zeros_like(image_raw)
    mask[:, ixs < outpaint_size] = 255
    mask[:, :, ixs < outpaint_size] = 255
    mask[:, ixs > initial_image.width + outpaint_size] = 255
    mask[:, :, ixs > initial_image.height + outpaint_size] = 255
    mask = _to_pillow(mask)

    image = generator(padded_image, mask)
    return image


def interp(
    src: Image.Image, dst: Image.Image, step: int, num_interps: int
) -> Animation:
    width, height = src.size
    # During outpainting we increased the image size by `step` on every side,
    # so now we need to know how much it is, after resizing to 512x512.
    inner_step = (step / (width + 2 * step)) * width

    frames = []
    for i in range(num_interps):
        canvas = Image.new("RGB", (width, height), (0, 0, 0))

        padding_src = round(i * (inner_step / num_interps))
        padding_dst = step - round(i * (step / num_interps))

        src_s = width - 2 * padding_src
        dst_s = width + 2 * padding_dst

        resized_src = src.resize((src_s, src_s))
        resized_dst = dst.resize((dst_s, dst_s))

        canvas.paste(resized_dst, (-padding_dst, -padding_dst))
        canvas.paste(resized_src, (padding_src, padding_src))

        frames.append(canvas)

    return frames


def animate(
    initial_image: Image.Image,
    generator: Generator,
    num_frames: int,
    outpaint_size: int,
    num_interp_steps: int,
    method: OutpaintMethod = outpaint_sequentially,
) -> Animation:
    resize_to_original = transforms.Resize(initial_image.size)
    image = initial_image
    frames = [image]
    for _ in range(num_frames):
        next_image = method(image, outpaint_size, generator)
        next_image = _to_pillow(resize_to_original(_to_tensor(next_image)))
        frames.append(next_image)
        image = next_image

    all_frames = []
    for fa, fb in zip(frames, frames[1:]):
        interps = interp(fa, fb, outpaint_size, num_interps=num_interp_steps)
        all_frames.extend(interps)
    all_frames.append(frames[-1])
    return all_frames
