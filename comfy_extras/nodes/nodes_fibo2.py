import math

import numpy as np
import torch
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

# briaai/Fibo-2.0-dev scheduler/scheduler_config.json
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15
# the model card's recipe for the full base model: num_inference_steps=50, guidance_scale=5.0
STEPS = 50


def calculate_shift(image_seq_len, base_seq_len=BASE_IMAGE_SEQ_LEN, max_seq_len=MAX_IMAGE_SEQ_LEN, base_shift=BASE_SHIFT,
                    max_shift=MAX_SHIFT):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def fibo2_sigmas(steps: int, width: int, height: int) -> torch.Tensor:
    """BriaFibo2Pipeline's schedule, as FlowMatchEulerDiscreteScheduler.set_timesteps computes it: sigmas from 1
    down to 1 / steps (float32), time-shifted exponentially by mu at the image's token count (2x2 patches of the
    1/8 latent), then 0."""
    if width % 16 or height % 16:
        raise ValueError(f"Fibo 2.0 sizes are multiples of 16, got {width}x{height}")
    mu = calculate_shift((height // 16) * (width // 16))
    sigmas = np.array(np.linspace(1.0, 1 / steps, steps)).astype(np.float32)
    sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1) ** 1.0)
    return torch.cat([torch.from_numpy(sigmas).to(torch.float32), torch.zeros(1)])


class Fibo2Scheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Fibo2Scheduler",
            category="model/sampling/schedulers",
            description="The Fibo 2.0 pipeline's schedule: steps sigmas from 1 to 1 / steps, shifted for the image size, then 0.",
            inputs=[
                io.Int.Input("steps", default=STEPS, min=1, max=4096),
                io.Int.Input("width", default=1024, min=16, max=16384, step=16),
                io.Int.Input("height", default=1024, min=16, max=16384, step=16),
            ],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, steps, width, height) -> io.NodeOutput:
        return io.NodeOutput(fibo2_sigmas(steps, width, height))


class Fibo2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Fibo2Scheduler]


async def comfy_entrypoint() -> Fibo2Extension:
    return Fibo2Extension()
