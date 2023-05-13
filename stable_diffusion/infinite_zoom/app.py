import itertools
import subprocess
import tempfile
from pathlib import Path

import gradio as gr
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from stable_diffusion.infinite_zoom.inpainting import (Animation, Generator, animate,
                                                       outpaint)

_LOADED_MODELS: dict[str, StableDiffusionInpaintPipeline] = {}
_DTYPE = torch.float16
if torch.cuda.is_available():
    _DEVICE = "cuda"
elif torch.has_mps:
    _DEVICE = "mps"
    _DTYPE = torch.float32
else:
    _DEVICE = "cpu"

_PIPE: StableDiffusionInpaintPipeline | None = None
_GENERATOR: torch.Generator | None = None
_PAINTER: Generator | None = None
_INITIAL_IMAGE: Image.Image | None = None
_FRAMES: Animation | None = None
_OUTPUT_FILE = "output.mp4"


def generate_initial_image(
    model: str,
    size: int,
    sampler: str,
    num_inference_steps: int,
    cfg: int,
    seed: int,
    prompt: str,
    negative_prompt: str,
) -> Image.Image:
    global _PIPE
    _PIPE = _LOADED_MODELS.get(model)
    if _PIPE is None:
        _PIPE = StableDiffusionInpaintPipeline.from_pretrained(
            model, torch_dtype=_DTYPE
        )
        _PIPE = _PIPE.to(_DEVICE)
        _PIPE.enable_attention_slicing()

    global _GENERATOR
    _GENERATOR = None
    if _DEVICE == "mps":
        # torch.Generator doesn't support MPS backend
        seed = -1
    if seed >= 0:
        _GENERATOR = torch.Generator(_DEVICE).manual_seed(seed)

    global _PAINTER
    _PAINTER = lambda image, mask: _PIPE(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=cfg,
        generator=_GENERATOR,
        height=image.height,
        width=image.width,
    ).images[0]

    blank_image = Image.new("RGB", (size, size), (0, 0, 0))
    mask = Image.new("RGB", (size, size), (255, 255, 255))

    global _INITIAL_IMAGE
    _INITIAL_IMAGE = _PAINTER(blank_image, mask)
    return _INITIAL_IMAGE


def generate_frames(
    num_zoom_steps: int,
    zoom_step_size: int,
) -> Animation:
    global _PAINTER, _INITIAL_IMAGE, _FRAMES
    _FRAMES = list(
        itertools.islice(
            outpaint(_INITIAL_IMAGE, _PAINTER, zoom_step_size), num_zoom_steps
        )
    )
    return _FRAMES


def generate_animation(
    zoom_step_size: int,
    num_zoom_step_interps: int,
    frame_duration: int,
) -> str:
    global _FRAMES
    animation = animate(_FRAMES, zoom_step_size, num_zoom_step_interps)
    fps = 1000 // frame_duration
    _export(animation, _OUTPUT_FILE, fps)
    return _OUTPUT_FILE


def _export(frames: list[Image.Image], output: str, fps: int):
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, frame in enumerate(frames):
            frame.save(Path(tmp_dir) / f"frame_{i:06d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-pattern_type",
                "glob",
                "-i",
                f"{tmp_dir}/frame_*.png",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                output,
            ],
            check=True,
        )


def interface() -> gr.Blocks:
    _MODELS = [
        "stabilityai/stable-diffusion-2-inpainting",
        "runwayml/stable-diffusion-inpainting",
    ]
    _SAMPLERS = [
        "WIP",
    ]

    with gr.Blocks(title="Infinite zoom") as root:
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    prompt = gr.Textbox(
                        label="Prompt",
                        lines=5,
                        value="space, cosmos, photorealistic, nebula, 8k",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative prompt",
                        lines=5,
                        value="frame, borderline, watermark, text, ugly, deformed",
                    )
                    model = gr.Dropdown(_MODELS, value=_MODELS[0], label="Model")
                    size = gr.Slider(32, 512, value=512, step=32, label="Image size")
                    sampler = gr.Dropdown(
                        _SAMPLERS, value=_SAMPLERS[0], label="Sampler"
                    )
                    num_inference_steps = gr.Slider(
                        1, 100, value=20, step=1, label="Inference steps"
                    )
                    cfg = gr.Slider(1, 50, step=0.5, value=7.5, label="CFG")
                    seed = gr.Slider(-1, 1_000_000, step=1, value=-1, label="Seed")
            with gr.Column():
                submit_initial_image = gr.Button("Generate initial image")
                initial_image = gr.Image(label="Initial image", type="pil")

        with gr.Row():
            with gr.Column():
                with gr.Box():
                    num_zoom_steps = gr.Slider(
                        1, 100, value=5, step=1, label="Zoom steps"
                    )
                    zoom_step_size = gr.Slider(
                        32, 128, value=128, step=32, label="Zoom step size"
                    )
            with gr.Column():
                submit_frames = gr.Button("Generate frames")
                frames = gr.Gallery(label="Generated frames")

        with gr.Row():
            with gr.Column():
                with gr.Box():
                    num_zoom_step_interps = gr.Slider(
                        2, 32, value=16, step=2, label="Zoom step interpolations"
                    )
                    frame_duration = gr.Slider(
                        1, 1000, value=33, step=1, label="Frame duration [ms]"
                    )
            with gr.Column():
                submit_animation = gr.Button("Generate animation")
                animation = gr.Video(label="Animation", interactive=False)

        submit_initial_image.click(
            generate_initial_image,
            inputs=[
                model,
                size,
                sampler,
                num_inference_steps,
                cfg,
                seed,
                prompt,
                negative_prompt,
            ],
            outputs=[initial_image],
        )
        submit_frames.click(
            generate_frames,
            inputs=[
                num_zoom_steps,
                zoom_step_size,
            ],
            outputs=[frames],
        )
        submit_animation.click(
            generate_animation,
            inputs=[
                zoom_step_size,
                num_zoom_step_interps,
                frame_duration,
            ],
            outputs=[animation],
        )
    return root


demo = interface().queue()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8888)
