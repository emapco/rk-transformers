from typing import Literal, get_args

RKNN_WEIGHTS_NAME = "model.rknn"
RKNN_FILE_PATTERN = r".*\.rknn$"

PlatformType = Literal[
    "simulator",
    "rk3588",
    "rk3576",
    "rk3568",
    "rk3566",
    "rk3562",
    "rv1126b",
    "rv1106b",
    "rv1106",
    "rv1103b",
    "rv1103",
    "filled_target_platform",
    "support_target_platform",
]
CoreMaskType = Literal[  # for rk3588/rk3576 devices or other device with multiple NPU cores
    "auto",
    "0",
    "1",
    "2",
    "0_1",
    "0_1_2",  # only supported by RK3588
    "all",
]
PLATFORM_CHOICES = get_args(PlatformType)
CORE_MASK_CHOICES = get_args(CoreMaskType)
