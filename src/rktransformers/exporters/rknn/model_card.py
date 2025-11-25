# Copyright 2025 Emmanuel Cortes. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os

from huggingface_hub import ModelCard, ModelCardData

from rktransformers.configuration import RKNNConfig
from rktransformers.exporters.rknn.utils import get_file_size_str, get_rk_model_class
from rktransformers.utils.env_utils import get_rknn_toolkit_version, get_rktransformers_version
from rktransformers.utils.modeling_utils import check_sentence_transformer_support

logger = logging.getLogger(__name__)


class ModelCardGenerator:
    """Generates a README.md model card for RKNN exported models."""

    def __init__(self, template_path: str | None = None):
        if template_path is None:
            template_path = os.path.join(os.path.dirname(__file__), "model_card_template.md")

        self.template_path = template_path

    def _get_available_models(self, output_dir: str, config: RKNNConfig | None = None) -> list[RKNNConfig]:
        """
        Get all available RKNN models in the output directory with their configurations.

        Args:
            output_dir: Directory containing RKNN models.
            config: Optional RKNN config to use for inferring model details when rknn.json is missing.

        Returns:
            List of RKNNConfig objects, one for each available RKNN model.
            Each config includes the relative path in output_path field.
        """
        available_configs = []
        # Load rknn.json if it exists
        rknn_json_path = os.path.join(output_dir, "rknn.json")
        rknn_configs = {}
        if os.path.exists(rknn_json_path):
            try:
                with open(rknn_json_path) as f:
                    rknn_configs = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load rknn.json: {e}")

        # If rknn.json exists, use it as the source of truth, but verify files exist
        if rknn_configs:
            for rel_path, model_config_dict in rknn_configs.items():
                full_path = os.path.join(output_dir, rel_path)
                if os.path.exists(full_path):
                    # Load config using from_dict for proper deserialization
                    try:
                        model_config = RKNNConfig.from_dict(model_config_dict)
                        # Ensure output_path is set to the relative path for reference
                        model_config.output_path = rel_path
                        available_configs.append(model_config)
                    except Exception as e:
                        logger.warning(f"Failed to load config for {rel_path}: {e}")
        else:
            # Fallback to scanning directory if rknn.json is missing
            logger.warning("rknn.json not found, falling back to directory scanning")
            # Check root directory
            for f in os.listdir(output_dir):
                if f.endswith(".rknn"):
                    # Create minimal config, using provided config as base if available
                    if config and config.output_path and f == os.path.basename(config.output_path):
                        # Clone the provided config and update output_path
                        model_config = RKNNConfig.from_dict(config.to_export_dict())
                        model_config.output_path = f
                    else:
                        # Create minimal config with unknown optimization/quantization
                        model_config = RKNNConfig(
                            output_path=f,
                            target_platform=config.target_platform if config else "rk3588",
                        )
                    available_configs.append(model_config)

            # Check rknn subdirectory
            rknn_subdir = os.path.join(output_dir, "rknn")
            if os.path.exists(rknn_subdir) and os.path.isdir(rknn_subdir):
                for f in os.listdir(rknn_subdir):
                    if f.endswith(".rknn"):
                        path = f"rknn/{f}"
                        # Create minimal config, using provided config as base if available
                        if (
                            config
                            and config.output_path
                            and os.path.abspath(os.path.join(output_dir, path)) == os.path.abspath(config.output_path)
                        ):
                            # Clone the provided config and update output_path
                            model_config = RKNNConfig.from_dict(config.to_export_dict())
                            model_config.output_path = path
                        else:
                            # Create minimal config with unknown optimization/quantization
                            model_config = RKNNConfig(
                                output_path=path,
                                target_platform=config.target_platform if config else "rk3588",
                            )
                        available_configs.append(model_config)
        return available_configs

    def _resolve_model_name(self, config: RKNNConfig) -> str:
        """
        Resolve the model name from the configuration.

        Args:
            config: RKNN configuration.

        Returns:
            Model name extracted from model_id_or_path.
        """
        assert config.model_id_or_path is not None, (
            "model_id_or_path should be set in RKNNConfig before generating the model card"
        )
        if os.path.exists(config.model_id_or_path):
            # Local file/directory
            abs_path = os.path.abspath(config.model_id_or_path)
            if os.path.isfile(abs_path):
                return os.path.basename(os.path.dirname(abs_path))
            else:
                return os.path.basename(abs_path)
        else:
            # Hub ID
            if "/" in config.model_id_or_path:
                return config.model_id_or_path.split("/")[-1]
            else:
                return config.model_id_or_path

    def _resolve_base_model_info(self, config: RKNNConfig, base_model_id: str | None) -> tuple[str, str]:
        """
        Resolve the base model and its URL.

        Args:
            config: RKNN configuration.
            base_model_id: Optional base model ID override.

        Returns:
            Tuple of (base_model, base_model_url).
        """
        assert config.model_id_or_path is not None, (
            "model_id_or_path should be set in RKNNConfig before generating the model card"
        )
        base_model = base_model_id if base_model_id else config.model_id_or_path

        if os.path.exists(base_model):
            # Local file/directory
            if os.path.isfile(base_model):
                base_model = os.path.dirname(os.path.abspath(base_model))
            else:
                base_model = os.path.abspath(base_model)
            base_model_url = base_model
        else:
            # Assume Hub ID
            base_model_url = f"https://huggingface.co/{base_model}"

        return base_model, base_model_url

    def _prepare_example_paths(
        self, config: RKNNConfig, default_model_path: str, model_name: str
    ) -> tuple[str, str | None, str]:
        """
        Prepare paths for usage examples in the model card.

        Args:
            config: RKNN configuration.
            default_model_path: Default model file path.
            model_name: Resolved model name.

        Returns:
            Tuple of (example_model_path, example_file_name, tokenizer_path).
        """
        # Priority 1: Use hub_model_id if provided
        if config.hub_model_id:
            example_model_path = config.hub_model_id
            tokenizer_path = config.hub_model_id
        else:
            # Priority 2: Local path - use the model name
            example_model_path = model_name
            tokenizer_path = model_name

        # If the default model is NOT model.rknn, we need to specify file_name
        example_file_name = default_model_path if default_model_path != "model.rknn" else None

        return example_model_path, example_file_name, tokenizer_path

    def _merge_with_existing_readme(self, rknn_card: ModelCard, card_data: ModelCardData, readme_path: str) -> None:
        """
        Merge RKNN card with existing README if present.

        Args:
            rknn_card: The RKNN model card to merge.
            card_data: The card data to use for merging.
            readme_path: Path to the README file.
        """
        try:
            existing_card = ModelCard.load(readme_path)

            # Check if it's already an rk-transformers card
            is_rktransformers_card = getattr(existing_card.data, "library_name", "") == "rk-transformers"

            if is_rktransformers_card:
                # It is already an rk-transformers card, just overwrite/update
                rknn_card.save(readme_path)
                return

            # Merge metadata
            if existing_card.data:
                # Replace specific fields
                existing_card.data.model_name = card_data.model_name
                existing_card.data.base_model = card_data.base_model
                existing_card.data.library_name = card_data.library_name
                # Remove model-index from original model as it's not applicable to RKNN
                if getattr(existing_card.data, "eval_results", None) is not None:
                    existing_card.data.eval_results = None
                if getattr(existing_card.data, "model_index", None) is not None:
                    existing_card.data.model_index = None
                # Append tags
                if card_data.tags:
                    if existing_card.data.tags:
                        # Add only new tags
                        for tag in card_data.tags:
                            if tag not in existing_card.data.tags:
                                existing_card.data.tags.append(tag)
                    else:
                        existing_card.data.tags = card_data.tags
                # Replace datasets if quantization dataset is provided
                if card_data.datasets:
                    existing_card.data.datasets = card_data.datasets
                else:
                    existing_card.data.datasets = None
            else:
                existing_card.data = card_data

            # Handle content wrapping
            # Split RKNN card content into Header/Intro and Details
            rknn_content = rknn_card.text
            split_marker = "## Model Details"

            if split_marker in rknn_content:
                parts = rknn_content.split(split_marker, 1)
                header_part = parts[0].strip()
                details_part = split_marker + parts[1]
                new_content = (
                    f"{header_part}\n\n"
                    f"<details><summary>Click to see the RKNN model details and usage examples</summary>\n\n"
                    f"{details_part}\n\n"
                    f"</details>\n"
                    f"{existing_card.text}"
                )

                existing_card.text = new_content
            else:
                # Fallback if marker not found (shouldn't happen with standard template)
                existing_card.text = rknn_card.text + "\n\n" + existing_card.text

            existing_card.save(readme_path)

        except Exception as e:
            logger.warning(f"Failed to load/merge existing README.md: {e}. Overwriting.")
            rknn_card.save(readme_path)

    def generate(
        self,
        config: RKNNConfig,
        output_dir: str,
        base_model_id: str | None = None,
    ) -> str | None:
        """
        Generate the model card content and write it to the output directory.

        Args:
            config: The RKNN configuration used for export.
            output_dir: The directory where the model card should be saved.
            base_model_id: The original model ID or path (before export).

        Returns:
            The path to the generated model card file, or None if generation failed.
        """
        # Get RKNN toolkit version
        try:
            rknn_version_str = get_rknn_toolkit_version()
            rknn_version = rknn_version_str.split("==")[1] if "==" in rknn_version_str else rknn_version_str
        except Exception:
            rknn_version = "Unknown"

        assert config.model_id_or_path is not None, (
            "model_id_or_path should be set in RKNNConfig before generating the model card"
        )
        assert config.model_id_or_path is not None, (
            "output_path should be set in RKNNConfig before generating the model card"
        )

        model_name = self._resolve_model_name(config)
        base_model, base_model_url = self._resolve_base_model_info(config, base_model_id)

        datasets = []
        if config.quantization.dataset_name:
            datasets.append(config.quantization.dataset_name)

        is_sentence_transformer = check_sentence_transformer_support(output_dir, config.model_id_or_path)

        rk_model_class = get_rk_model_class(config.task)

        available_model_configs = self._get_available_models(output_dir, config)
        available_model_configs.sort(key=lambda x: x.output_path)  # type: ignore
        rknn_sub_models = [m.output_path for m in available_model_configs if "rknn/" in m.output_path]  # type: ignore
        rknn_models = [m.output_path for m in available_model_configs]

        # Select best optimized model from rknn subdirectory
        optimized_model_path = None
        if rknn_sub_models:
            priorities = ["w8a8", "o3", "o2", "o1"]
            for p in priorities:
                for m in rknn_sub_models:
                    if p in m:  # type: ignore
                        optimized_model_path = m
                        break
                if optimized_model_path:
                    break
            # Fallback to first available if no priority match
            if not optimized_model_path:
                rknn_sub_models.sort()  # type: ignore
                optimized_model_path = rknn_sub_models[0]

        rknn_models.sort()  # type: ignore
        default_model_path = (
            rknn_models[0]
            if rknn_models
            else (os.path.basename(config.output_path) if config.output_path else "model.rknn")
        )
        assert default_model_path is not None

        card_data = ModelCardData(
            model_name=model_name,
            base_model=base_model,
            tags=["rknn", "rockchip", "npu", "rk-transformers", config.target_platform],
            datasets=datasets if datasets else None,
            library_name="rk-transformers",
        )

        example_model_path, example_file_name, tokenizer_path = self._prepare_example_paths(
            config, default_model_path, model_name
        )

        available_models = [
            {
                "path": model_config.output_path,
                "optimization_level": model_config.optimization.optimization_level,
                "quantization": (
                    model_config.quantization.quantized_dtype
                    if model_config.quantization.do_quantization
                    else "float16"
                ),
                "file_size": get_file_size_str(os.path.join(output_dir, model_config.output_path)),  # type: ignore
            }
            for model_config in available_model_configs
        ]

        rknn_card = ModelCard.from_template(
            card_data,
            template_path=self.template_path,
            model_name=model_name,
            base_model=base_model,
            base_model_url=base_model_url,
            datasets=datasets,
            is_sentence_transformer=is_sentence_transformer,
            rk_model_class=rk_model_class,
            target_platform=config.target_platform,
            quantized=config.quantization.do_quantization,
            quantized_dtype=config.quantization.quantized_dtype if config.quantization.do_quantization else "None",
            rknn_version=rknn_version,
            rk_transformers_version=get_rktransformers_version(),
            output_path=default_model_path,
            optimized_model_path=optimized_model_path,
            example_model_path=example_model_path,
            example_file_name=example_file_name,
            tokenizer_path=tokenizer_path,
            available_models=available_models,
            max_seq_length=config.max_seq_length,
        )

        if getattr(rknn_card.data, "eval_results", None) is not None:
            rknn_card.data.eval_results = None
        if getattr(rknn_card.data, "model_index", None) is not None:
            rknn_card.data.model_index = None

        readme_path = os.path.join(output_dir, "README.md")

        if not os.path.exists(readme_path):
            rknn_card.save(readme_path)
            return readme_path

        self._merge_with_existing_readme(rknn_card, card_data, readme_path)
        return readme_path
