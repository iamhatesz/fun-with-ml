import gradio as gr
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from torchvision.transforms import transforms

_LOADED_MODELS = {}

if torch.cuda.is_available():
    _DEVICE = "cuda"
elif torch.has_mps:
    _DEVICE = "mps"
else:
    _DEVICE = "cpu"

_to_pillow = transforms.ToPILImage()
_to_tensor = transforms.PILToTensor()


def generate(
    model: str,
    size: int,
    sampler: str,
    num_inference_steps: int,
    cfg: int,
    seed: int,
    num_zoom_steps: int,
    zoom_step_size: int,
    num_zoom_step_interps: int,
    frame_duration: int,
    prompt: str,
    negative_prompt: str,
) -> str:
    pipe = _LOADED_MODELS.get(model)
    if pipe is None:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model, torch_dtype=torch.float16
        )
        pipe = pipe.to(_DEVICE)
        pipe.enable_attention_slicing()

    generator = None
    if seed >= 0:
        generator = torch.Generator(_DEVICE).manual_seed(seed)

    blank_image = Image.new("RGB", (size, size), (0, 0, 0))
    mask = Image.new("RGB", (size, size), (255, 255, 255))

    initial_image = image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=blank_image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=cfg,
        generator=generator,
    ).images[0]

    step_transform = transforms.Compose(
        [
            transforms.Resize(size - 2 * zoom_step_size),
            transforms.Pad(zoom_step_size, padding_mode="edge"),
        ]
    )

    masks = []
    idx = torch.arange(size)

    mask_top = torch.zeros((3, size, size))
    mask_top[:, idx < zoom_step_size] = 1
    mask_top[:, :, idx < zoom_step_size] = 0
    mask_top[:, :, idx > size - zoom_step_size] = 0
    masks.append(_to_pillow(mask_top))

    mask_left = torch.zeros((3, size, size))
    mask_left[:, :, idx < zoom_step_size] = 1
    masks.append(_to_pillow(mask_left))

    mask_bottom = torch.zeros((3, size, size))
    mask_bottom[:, idx > size - zoom_step_size] = 1
    mask_bottom[:, :, idx < zoom_step_size] = 0
    mask_bottom[:, :, idx > size - zoom_step_size] = 0
    masks.append(_to_pillow(mask_bottom))

    mask_right = torch.zeros((3, size, size))
    mask_right[:, :, idx > size - zoom_step_size] = 1
    masks.append(_to_pillow(mask_right))

    # zoom_out_mask = torch.zeros((3, size, size))
    # zoom_out_mask[:, idx < zoom_step_size] = 1
    # zoom_out_mask[:, :, idx < zoom_step_size] = 0
    # zoom_out_mask[:, :, idx > size - zoom_step_size] = 0
    # # zoom_out_mask[:, idx > size - zoom_step_size] = 1
    # # zoom_out_mask[:, :, idx < zoom_step_size] = 1
    # # zoom_out_mask[:, :, idx > size - zoom_step_size] = 1
    # zoom_out_mask = _to_pillow(zoom_out_mask)

    frames: list[Image.Image] = []
    for _ in range(num_zoom_steps):
        zoomed_out_image = _to_pillow(step_transform(_to_tensor(image)))

        for mask in masks:
            next_image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=zoomed_out_image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=cfg,
                generator=generator,
            ).images[0]
            zoomed_out_image = next_image

        new_frames = _interp(
            image,
            next_image,
            step=zoom_step_size,
            num_interps=num_zoom_step_interps,
        )
        frames.extend(new_frames)
        image = next_image
        # break

    frames[0].save(
        "output.gif",
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=frame_duration,
        loop=0,
    )
    return initial_image, masks[-1], "output.gif"
    # return initial_image, masks[-1], frames[-1]


def _interp(
    src: Image.Image, dst: Image.Image, step: int, num_interps: int
) -> list[Image.Image]:
    width, height = src.size

    # 0 => 512 / ?
    # 1 => 512 - 2 * (1 * 32) / 512 + 2 * (3 * 32)
    # 2 => 512 - 2 * (2 * 32) / 512 + 2 * (2 * 32)
    # 3 => 512 - 2 * (3 * 32) / 512 + 2 * (1 * 32)
    # 4 => 512 - 2 * (4 * 32) / 512 + 2 * (0 * 32)

    frames = []

    for i in range(1, num_interps + 1):
        canvas = Image.new("RGB", (width, height), (0, 0, 0))

        padding_src = i * (step // num_interps)
        padding_dst = -2 * ((num_interps - i) * (step // num_interps))

        src_s = width - 2 * padding_src
        dst_s = width - 2 * padding_dst

        resized_src = src.resize((src_s, src_s))
        resized_dst = dst.resize((dst_s, dst_s))

        canvas.paste(resized_dst, (padding_dst, padding_dst))
        canvas.paste(resized_src, (padding_src, padding_src))

        frames.append(canvas)

    return frames


def interface() -> gr.Blocks:
    _MODELS = [
        "stabilityai/stable-diffusion-2-inpainting",
        "runwayml/stable-diffusion-inpainting",
    ]
    _SAMPLERS = [
        "asd",
    ]

    with gr.Blocks(title="Infinite zoom") as root:
        gr.Markdown("Hello!")
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    prompt = gr.Textbox(label="Prompt", lines=5)
                    negative_prompt = gr.Textbox(label="Negative prompt", lines=5)
                with gr.Box():
                    num_zoom_steps = gr.Slider(
                        1, 100, value=10, step=1, label="Zoom steps"
                    )
                    zoom_step_size = gr.Slider(
                        1, 256, value=128, step=1, label="Zoom step size"
                    )
                    num_zoom_step_interps = gr.Slider(
                        2, 64, value=16, step=1, label="Zoom step interpolations"
                    )
                    frame_duration = gr.Slider(
                        1, 1000, value=50, step=1, label="Frame duration [ms]"
                    )
                submit = gr.Button("Generate")
                submit.style(size="lg")
            with gr.Column():
                with gr.Box():
                    model = gr.Dropdown(_MODELS, value=_MODELS[0], label="Model")
                    size = gr.Slider(32, 512, value=512, step=1, label="Image size")
                    sampler = gr.Dropdown(
                        _SAMPLERS, value=_SAMPLERS[0], label="Sampler"
                    )
                    num_inference_steps = gr.Slider(
                        1, 100, value=20, step=1, label="Inference steps"
                    )
                    cfg = gr.Slider(1, 50, step=0.5, value=7.5, label="CFG")
                    seed = gr.Slider(-1, 1_000_000, step=1, value=-1, label="Seed")
        with gr.Row():
            with gr.Column():
                initial_image = gr.Image(label="Initial image")
            with gr.Column():
                mask = gr.Image(label="Inpainting mask")
            with gr.Column():
                output = gr.Image(label="Output")

        submit.click(
            generate,
            inputs=[
                model,
                size,
                sampler,
                num_inference_steps,
                cfg,
                seed,
                num_zoom_steps,
                zoom_step_size,
                num_zoom_step_interps,
                frame_duration,
                prompt,
                negative_prompt,
            ],
            outputs=[initial_image, mask, output],
        )
    return root


if __name__ == "__main__":
    root = interface()
    root.launch(server_name="0.0.0.0", server_port=8888)
