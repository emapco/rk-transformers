#!/usr/bin/env python3
#
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

"""Utility script to load RKNN2 models with rkruntime.RKRTModel."""

import argparse
import logging
from pathlib import Path
from typing import Any

from rkruntime.constants import PLATFORM_CHOICES
from rkruntime.modeling import RKRTModel

LOGGER = logging.getLogger("load_hf_rknn")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and initialize an RKNN model from the Hugging Face Hub using RKRTModel."
    )
    parser.add_argument(
        "--model-id",
        default="happyme531/MeloTTS-RKNN2",
        help="Repository ID or local path pointing to the RKNN assets.",
    )
    parser.add_argument(
        "--platform",
        choices=PLATFORM_CHOICES,
        default=None,
        help="Target platform for RKNN runtime initialization. Auto-detected by default.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision to fetch (branch, tag, or commit).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for downloaded artifacts.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of artifacts from the Hub.",
    )
    parser.add_argument(
        "--resume-download",
        action="store_true",
        help="Resume a previously interrupted download.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached files without attempting network downloads.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of custom code from the repository when loading configs.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token to access private repositories, if required.",
    )
    parser.add_argument(
        "--file-name",
        default=None,
        help="Specify the RKNN file name if the the wrong model file is auto-detected.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save a local copy of the model once loaded.",
    )
    return parser


def load_model(args: argparse.Namespace) -> Any:
    LOGGER.info("Loading RKNN model from %s", args.model_id)
    model = RKRTModel.from_pretrained(
        pretrained_model_name_or_path=args.model_id,
        # rknn options
        platform=args.platform,
        # hub options
        revision=args.revision,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
        resume_download=args.resume_download,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
        # file options
        file_name=args.file_name,
    )
    LOGGER.info("Loaded RKNN from %s", model.model_path)
    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    model = load_model(args)

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_dir)
        LOGGER.info("Saved RKNN assets to %s", args.save_dir)

    LOGGER.info(
        "Model ready (config: %s, platform: %s, core_mask: %s)",
        model.config.name_or_path,
        model.platform,
        model.core_mask,
    )


if __name__ == "__main__":
    main()
