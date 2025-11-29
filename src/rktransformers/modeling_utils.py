import contextlib
import sys
from abc import ABC
from pathlib import Path
from typing import Any, TypeVar

from transformers.utils import logging
from transformers.utils.generic import ModelOutput
from typing_extensions import TypeVarTuple

from .configuration import RKNNConfig
from .constants import (
    CoreMaskType,
    PlatformType,
)
from .utils.env_utils import get_edge_host_platform
from .utils.import_utils import (
    is_rknn_toolkit_available,
    is_rknn_toolkit_lite_available,
)
from .utils.logging_utils import suppress_output

logger = logging.get_logger(__name__)

if is_rknn_toolkit_lite_available():
    from rknnlite.api import RKNNLite  # pyright: ignore[reportMissingImports]
elif is_rknn_toolkit_available():
    # Fallback to RKNN if RKNNLite is not available. RKNN shares similar functionality as RKNNLite.
    from rknn.api import RKNN as RKNNLite  # pyright: ignore[reportMissingImports]
else:
    logger.error("RKNN Toolkit Lite is not installed. Please install it via pip:")
    logger.error("  pip install rknn-toolkit-lite2==2.3.2")
    sys.exit(-1)


MODEL_OUTPUT_T = TypeVar("MODEL_OUTPUT_T", bound=ModelOutput)
TENSOR_Ts = TypeVarTuple("TENSOR_Ts")


# workaround to enable compatibility between rk-transformers models and transformers pipelines
class PreTrainedModel(ABC):  # noqa: B024
    pass


class RKNNRuntime:
    """Handles RKNN model loading and runtime initialization."""

    def __init__(
        self,
        model_path: str | Path,
        platform: PlatformType | None = None,
        core_mask: CoreMaskType = "auto",
        rknn_config: RKNNConfig | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.platform = platform or get_edge_host_platform()
        self.core_mask = core_mask
        self.rknn_config = rknn_config
        self.rknn = self._load_rknn_model()

    def _release(self) -> None:
        """Release RKNNLite resources following RKNN-toolkit2 best practices."""
        if getattr(self, "rknn", None) is not None:
            assert self.rknn is not None
            with contextlib.suppress(Exception), suppress_output():
                self.rknn.release()
            self.rknn = None

    def __del__(self) -> None:  # pragma: no cover - destructor safety
        self._release()

    def list_model_compatible_platform(self) -> dict[str, Any] | None:
        """
        List supported platforms for the loaded model.
        Returns a dictionary of supported platforms or None if the check fails.
        """
        if self.rknn is None:
            return None
        with suppress_output():
            if hasattr(self.rknn, "list_support_target_platform") and callable(self.rknn.list_support_target_platform):  # type: ignore
                return self.rknn.list_support_target_platform(self.model_path.as_posix())  # type: ignore
        return None

    def _load_rknn_model(self) -> RKNNLite:
        """Load the RKNN graph and initialize the runtime."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"RKNN model not found: {self.model_path}")

        # Suppress RKNNLite C-level logging
        with suppress_output():
            rknn = RKNNLite(verbose=False)
            logger.debug("Loading RKNN model from %s", self.model_path)
            ret = rknn.load_rknn(self.model_path.as_posix())
        if ret != 0:
            raise RuntimeError("Failed to load RKNN model")

        # Check model-platform compatibility
        supported_platforms = None
        if hasattr(rknn, "list_support_target_platform") and callable(rknn.list_support_target_platform):  # type: ignore
            with suppress_output():
                supported_platforms: dict[str, Any] | None = rknn.list_support_target_platform(  # type: ignore
                    self.model_path.as_posix()
                )

        if supported_platforms is not None:
            # The return value is an OrderedDict with a key 'support_target_platform' and 'filled_target_platform'
            # containing a list of supported platforms.
            # Note: Only the 'support_target_platform' is relevant for compatibility check.
            support_platforms = [p.lower() for p in supported_platforms.get("support_target_platform", [])]
            if self.platform is not None and len(support_platforms) > 0 and self.platform not in support_platforms:
                raise RuntimeError(
                    f"The model is not compatible with the current platform '{self.platform}'. "
                    f"Supported platforms: {support_platforms}"
                )

        logger.debug(
            "Initializing RKNN runtime (platform=%s, core_mask=%s)",
            self.platform,
            self.core_mask,
        )
        if self.platform in {"rk3588", "rk3576"}:
            core_mask_map = {
                "auto": RKNNLite.NPU_CORE_AUTO,
                "0": RKNNLite.NPU_CORE_0,
                "1": RKNNLite.NPU_CORE_1,
                "2": RKNNLite.NPU_CORE_2,
                "0_1": RKNNLite.NPU_CORE_0_1,
                "0_1_2": RKNNLite.NPU_CORE_0_1_2,
                "all": RKNNLite.NPU_CORE_ALL,
            }
            npu_core = core_mask_map.get(self.core_mask, RKNNLite.NPU_CORE_AUTO)
            with suppress_output():
                ret = rknn.init_runtime(core_mask=npu_core)
                if ret != 0:
                    ret = rknn.init_runtime(target=self.platform, core_mask=npu_core)
        else:
            with suppress_output():
                ret = rknn.init_runtime()

        if ret != 0:
            raise RuntimeError("Failed to initialize RKNN runtime")

        return rknn
