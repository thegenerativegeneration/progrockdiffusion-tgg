import logging
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Union, List, Tuple

import torch

logger = logging.getLogger(__name__)

METRIC_LABELS: List[str] = ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
BINARY_LABELS: List[str] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]
PRECISION_OFFSETS: List[float] = [0.5, 0.05, 0.005, 0.0005]
PRECISION_FORMATS: List[str] = ["{}{:.0f} {}", "{}{:.1f} {}", "{}{:.2f} {}",
                                "{}{:.3f} {}"]
CUDA_MAX_ALLOCATED = 0
LOADED_MODEL_TOTAL_SIZE = 0

LPIPS_LOAD_SIZE = 59259904


@contextmanager
def track_model_vram(device, message=''):
    global LOADED_MODEL_TOTAL_SIZE
    initial_memory = torch.cuda.memory_allocated(device)
    try:
        yield
    finally:
        peak = torch.cuda.max_memory_allocated()
        peak_delta = peak - initial_memory
        LOADED_MODEL_TOTAL_SIZE += peak_delta
        logger.debug(f"{message}: {format_bytes(peak_delta)}")
        logger.debug(f"Cumulative loaded model size: {format_bytes(LOADED_MODEL_TOTAL_SIZE)}")


@contextmanager
def track_max_vram_allocated(device, message=''):
    """
    Use as a context manager to wrap a block of code and log the maximum relative and absolute
    VRAM allocated as that code is executed as well as the VRAM allocated when the code is
    started and finished.

    For example:

    ```
    with track_max_vram_allocated(device, "test_function"):
        # Some code you want to track
        if use_foo:
            use_lots_of_vram(foo)
        else:
            use_lots_of_vram(bar)
    ```

    NOTE: If you use this function, you MUST use `log_max_allocated()` or otherwise use
    of `max((CUDA_MAX_ALLOCATED, torch.cuda.max_memory_allocated(device)))`, rather than accessing
    cuda peak memory stats directly. This function has to reset the PyTorch peak memory stats in
    order to track a local maximum, and instead maintains the global maximum in the CUDA_MAX_ALLOCATED
    variable.
    """
    global CUDA_MAX_ALLOCATED
    CUDA_MAX_ALLOCATED = max((CUDA_MAX_ALLOCATED, torch.cuda.max_memory_allocated(device)))
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = torch.cuda.memory_allocated(device)
    try:
        yield
    finally:
        final_memory = torch.cuda.memory_allocated(device)
        max_absolute = torch.cuda.max_memory_allocated()
        max_relative = max_absolute - initial_memory
        logger.debug(f"{message}: initial {format_bytes(initial_memory)}")
        logger.debug(f"{message}: final {format_bytes(final_memory)}")
        logger.debug(f"{message}: relative max {format_bytes(max_relative)}")
        logger.debug(f"{message}: absolute max {format_bytes(max_absolute)}")


def log_vram(device, message=''):
    current_vram = torch.cuda.memory_allocated(device)
    logger.debug(f"{message}: current {format_bytes(current_vram)}")


def log_max_allocated(device):
    max_vram = max((CUDA_MAX_ALLOCATED, torch.cuda.max_memory_allocated(device)))
    logger.debug(f"Global max: {format_bytes(max_vram)}")


def format_bytes(num: Union[int, float], metric: bool = False, precision: int = 2, include_byte_int=True) -> str:
    """
    Human-readable formatting of bytes, using binary (powers of 1024)
    or metric (powers of 1000) representation.
    """

    assert isinstance(num, (int, float)), "num must be an int or float"
    assert isinstance(metric, bool), "metric must be a bool"
    assert isinstance(precision, int) and 3 >= precision >= 0, "precision must be an int (range 0-3)"

    bytes = num
    unit_labels = METRIC_LABELS if metric else BINARY_LABELS
    last_label = unit_labels[-1]
    unit_step = 1000 if metric else 1024
    unit_step_thresh = unit_step - PRECISION_OFFSETS[precision]

    is_negative = num < 0
    if is_negative: # Faster than ternary assignment or always running abs().
        num = abs(num)

    for unit in unit_labels:
        if num < unit_step_thresh:
            break
        if unit != last_label:
            num /= unit_step

    out_str = PRECISION_FORMATS[precision].format("-" if is_negative else "", num, unit)
    if include_byte_int:
        out_str += f"\t{bytes}"
    return out_str


@dataclass
class ClipModelProfile:
    '''
    Data class for storing CLIP model memory profile data and calculating estimated peak VRAM usage.
    '''
    name: str
    load_size: int = 0
    cut_coef: Tuple[Union[float, int], ...] = (0,)

    def estimate_peak(self, cuts):
        return int(sum(coef * cuts ** i for i, coef in enumerate(self.cut_coef)))

@dataclass
class DiffusionModelProfile:
    '''
    Data class for storing diffusion model memory profile data and calculating estimated peak VRAM usage.
    '''
    name: str
    weight_and_grad_size: int = 0
    init_px_coef: Tuple[Union[float, int], ...] = (0,)
    loss_px_coef: Tuple[Union[float, int], ...] = (0,)
    loss_correction_coef: Tuple[Union[float, int], ...] = (-4.13e06, 1.35)

    def estimate_init_bytes(self, pixels) -> int:
        return int(sum(coef * pixels ** i for i, coef in enumerate(self.init_px_coef)))

    def estimate_loss_bytes(self, pixels) -> int:
        reported_size = int(sum(coef * pixels ** i for i, coef in enumerate(self.loss_px_coef)))
        # The reported allocated VRAM during this step is consistently less than final
        # global maximum, even when this step seems to be the memory bottleneck for the
        # run. Rather than actually figure out where this additional memory is allocated,
        # this code assumes the existence of dark VRAM and factors it into the model
        # with the loss_correction_coef.
        corrected_size = int(sum(coef * reported_size ** i for i, coef in enumerate(self.loss_correction_coef)))
        return corrected_size
        # return reported_size


_512x512_diffusion_uncond_finetune_008100 = DiffusionModelProfile(
    name='512x512_diffusion_uncond_finetune_008100',
    weight_and_grad_size=1242808320,
    init_px_coef=(5.04e06, 6358.0),
    loss_px_coef=(9.9e06, 4614.0)
)

_512x512_diffusion_uncond_finetune_008100_with_secondary = DiffusionModelProfile(
    name='512x512_diffusion_uncond_finetune_008100_with_secondary',
    weight_and_grad_size=1242808320 + 57203712,  # diffusion with weights and grad + secondary
    init_px_coef=(5e06, 2601),
    loss_px_coef=(-4.6e07, 1399, -4.25e-05),
)

RN101 = ClipModelProfile(
    name='RN101',
    load_size=294541824,
    cut_coef=(94069930, 86131029)
)

RN50 = ClipModelProfile(
    name='RN50',
    load_size=256350208,
    cut_coef=(72989184, 65367040)
)

RN50x16 = ClipModelProfile(
    name='RN50x16',
    load_size=679183360,
    cut_coef=(463826432, 449804288)
)

RN50x4 = ClipModelProfile(
    name='RN50x4',
    load_size=425698304,
    cut_coef=(166758058, 164629845)
)

RN50x64 = ClipModelProfile(
    name='RN50x64',
    load_size=1369234944,
    cut_coef=(984844117, 1037487787)
)

ViTB16 = ClipModelProfile(
    name='ViTB16',
    load_size=357165568,
    cut_coef=(102521685, 92044458)
)

ViTB32 = ClipModelProfile(
    name='ViTB32',
    load_size=361563648,
    cut_coef=(35860309, 31099562)
)

ViTL14 = ClipModelProfile(
    name='ViTL14',
    load_size=942509568,
    cut_coef=(0, 0)
)

ViTL14_336 = ClipModelProfile(
    name='ViTL14_336',
    load_size=944606720,
    cut_coef=(0, 0)
)

CLIP_PROFILES = {
    model.name: model for model in [RN101, RN50, RN50x16, RN50x4, RN50x64, ViTB32, ViTB16, ViTL14, ViTL14_336]
}

DIFFUSION_PROFILES = {
    model.name: model for model in [
        _512x512_diffusion_uncond_finetune_008100,
        _512x512_diffusion_uncond_finetune_008100_with_secondary
    ]
}


def estimate_vram_requirements(
        side_x,
        side_y,
        cut_innercut,
        cut_overview,
        clip_model_names,
        diffusion_model_name,
        use_secondary,
        device
):
    """
    Estimate peak VRAM requirement, which is calculated as follows:

    The sum of the following:
      * size of all enabled CLIP models
      * size of LPIPS model
      * size of secondary diffusion model if enabled
      * size of diffusion model with enabled gradients
      * size of diffusion model initialization as a function of pixel count
      * size of prompt text embeddings (currently on the order of a few MiB and not included in estimates)
    Plus the maximum of the following:
      * maximum of CLIP model step allocations, which are a function of cuts
      * diffusion model loss step as a function of pixel count
    """
    if use_secondary:
        diffusion_model_name += '_with_secondary'
    diffusion_profile = DIFFUSION_PROFILES[diffusion_model_name]
    max_cuts = max(sum(x) for x in zip(eval(cut_innercut), eval(cut_overview)))

    static_sizes = {
        model_name: CLIP_PROFILES[model_name].load_size
        for model_name in clip_model_names
    }
    static_sizes['LPIPS'] = LPIPS_LOAD_SIZE
    static_sizes[diffusion_model_name] = diffusion_profile.weight_and_grad_size
    static_sizes['diffusion initialization'] = diffusion_profile.estimate_init_bytes(side_x * side_y)

    logger.debug("\tSTATIC ALLOCATION ESTIMATES (Allocated and kept until the end of the run)")
    for k, v in static_sizes.items():
        logger.debug(
            f"\t{format_bytes(v)}\t{k}"
        )
    static_sum = sum(static_sizes.values())
    logger.debug('')
    logger.debug(f"\t{format_bytes(static_sum)}\tTOTAL")
    logger.debug('')

    dynamic_sizes = {
        model_name: CLIP_PROFILES[model_name].estimate_peak(max_cuts)
        for model_name in clip_model_names
    }
    estimated_loss_vram = DIFFUSION_PROFILES[diffusion_model_name].estimate_loss_bytes(side_x * side_y)
    dynamic_sizes[diffusion_model_name + ' loss calculations'] = estimated_loss_vram

    logger.debug("\tDYNAMIC ALLOCATION ESTIMATES (Released after use during each step)")
    for k, v in dynamic_sizes.items():
        logger.debug(
            f"\t{format_bytes(v)}\t{k}"
        )
    dynamic_max = max(dynamic_sizes.values())
    logger.debug('')
    logger.debug(f"\t{format_bytes(dynamic_max)}\tMAX")

    logger.debug('')
    logger.debug("\tESTIMATED PEAK ALLOCATION (static total + dynamic max):")
    logger.debug(f"\t{format_bytes(static_sum + dynamic_max)}")
