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
    """Runtime wrapper for RKNN models.

    This class encapsulates loading an RKNN model, verifying its
    target device/platform compatibility, and initializing the runtime with the
    desired core mask.

    Attributes:
        model_path (Path): Filesystem path to the RKNN model file.
        platform (PlatformType | None): Target platform string such as ``'rk3588'``.
            When None, the platform is detected from the host environment.
        core_mask (CoreMaskType): Core mask selection for devices with multiple NPU
            cores. Examples include ``'auto'``, ``'0'``, ``'1'`, and ``'all'``.
        rknn_config (RKNNConfig | None): Optional configuration object for RKNN
            runtime behavior.
        rknn (RKNNLite | None): Loaded RKNN runtime instance or ``None`` when not
            initialized.

    Example:
        >>> runtime = RKNNRuntime("/tmp/model.rknn", platform="rk3588", core_mask="auto")
        >>> runtime.rknn  # The underlying RKNN runtime instance

    """

    def __init__(
        self,
        model_path: str | Path,
        platform: PlatformType | None = None,
        core_mask: CoreMaskType = "auto",
        rknn_config: RKNNConfig | None = None,
    ) -> None:
        """Create a new :class:`~RKNNRuntime` and loads the model specified by ``model_path``.

        Args:
            model_path (str | Path): Path to the RKNN model file on disk. This file
                will be loaded during initialization.
            platform (PlatformType | None, optional): Optional platform string
                specifying the target device. When None, the platform will be
                detected from the host environment via :py:func:`~rktransformers.utils.env_utils.get_edge_host_platform()`.
            core_mask (CoreMaskType, optional): Core mask used for devices with
                several NPU cores (e.g., 'auto', '0', '1', 'all'). Defaults to
                ``'auto'``.
            rknn_config (RKNNConfig | None, optional): Optional RKNN configuration
                object. Not all runtime options are currently implemented; this
                field is kept for future extension.

        Raises:
            FileNotFoundError: If the given `model_path` does not exist.
            RuntimeError: If the model fails to load or the runtime fails to
                initialize.
        """  # noqa: E501
        self.model_path = Path(model_path)
        self.platform = platform or get_edge_host_platform()
        self.core_mask = core_mask
        self.rknn_config = rknn_config
        self.rknn = self._load_rknn_model()

    def _release(self) -> None:
        """Release RKNN runtime resources.

        This method attempts to safely release the underlying RKNN runtime
        instance and suppresses exceptions to avoid propagating errors during
        interpreter shutdown. It also silences RKNN C-level logging while
        calling :meth:`rknn.release`.

        It is intentionally a private helper to mirror RKNN toolkit best-practices.
        """
        if getattr(self, "rknn", None) is not None:
            assert self.rknn is not None
            with contextlib.suppress(Exception), suppress_output():
                self.rknn.release()
            self.rknn = None

    def __del__(self) -> None:  # pragma: no cover - destructor safety
        self._release()

    def list_model_compatible_platform(self) -> dict[str, Any] | None:
        """Return the platforms supported by the current RKNN model.

        Returns:
            dict[str, Any] | None: The value returned by RKNN's
                ``list_support_target_platform`` helper or ``None`` if the runtime
                is not initialized or the API is not available.
                Example::

                    {
                        'support_target_platform': ['rk3588'],
                        'filled_target_platform': ['rk3588']
                    }
        """
        if self.rknn is None:
            return None
        with suppress_output():
            if hasattr(self.rknn, "list_support_target_platform") and callable(self.rknn.list_support_target_platform):  # type: ignore
                return self.rknn.list_support_target_platform(self.model_path.as_posix())  # type: ignore
        return None

    def _load_rknn_model(self) -> RKNNLite:
        """Load and initialize the RKNN model and runtime instance.

        This method performs the following steps:
        1. Verifies the model file exists and raises :class:`FileNotFoundError` if not.
        2. Creates an ``RKNNLite`` instance and loads the model.
        3. Optionally queries supported platforms and verifies compatibility.
        4. Initializes the runtime for the platform, applying ``core_mask`` on
           devices which expose multiple NPU cores.

        Returns:
            RKNNLite: Fully initialised RKNN runtime instance.

        Raises:
            FileNotFoundError: If the provided model file does not exist.
            RuntimeError: If the model fails to load or the runtime fails to
                initialize.
        """
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
