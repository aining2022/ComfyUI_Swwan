# SPDX-License-Identifier: GPL-3.0-only
# Image-node algorithms adapted from ComfyUI-KJNodes; see THIRD_PARTY_NOTICES.md.
import numpy as np
import time
import torch
import torch.nn.functional as F
import io
import base64
import random
import math
import os
import re
import json
import importlib
from pathlib import Path
from PIL.PngImagePlugin import PngInfo
from io import BytesIO
from PIL import ImageGrab, ImageDraw, ImageFont, Image, ImageOps
from nodes import MAX_RESOLUTION, SaveImage
import node_helpers
from comfy.cli_args import args
from comfy.utils import ProgressBar, common_upscale
import folder_paths
from comfy import model_management
from server import PromptServer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .dependencies import lazy_module, lazy_function
composite = lazy_function("comfy_extras.nodes_mask", "composite", "vision")
cv2 = lazy_module("cv2", "vision")
script_directory = str(Path(__file__).resolve().parents[1])
