from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
)
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings

from rtpt import RTPT


class LlavaOnevisionModel:

    # model: lmms-lab/llava-onevision-qwen2-72b-ov-chat

    def __init__(self):
        self.pretrained = "lmms-lab/llava-onevision-qwen2-72b-ov-chat"
        self.model_name = "llava_qwen"
        self.device = "cuda"
        self.device_map = "auto"  # "auto"
        self.max_tokens = 2048

        self.rtpt = RTPT(
            name_initials="XX", experiment_name="LLaVA-Onevision", max_iterations=10
        )
        self.rtpt.start()

        warnings.filterwarnings("ignore")

        print("Loading Llava-Onevision model...")
        self.tokenizer, self.model, self.image_processor, self.max_length = (
            load_pretrained_model(
                self.pretrained, None, self.model_name, device_map=self.device_map
            )
        )
        print("Model loaded.")

        self.model.eval()

    def prompt_with_images(
        self, prompt_text, image_paths, system_prompt=None, seed=None
    ):

        images = [Image.open(path) for path in image_paths]
        image_tensors = process_images(images, self.image_processor, self.model.config)
        image_tensors = [
            _image.to(dtype=torch.float16, device=self.device)
            for _image in image_tensors
        ]

        conv_template = "qwen_2"  # "chatml-llava"  # Make sure you use correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + prompt_text
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )
        image_sizes = [image.size for image in images]

        cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            # do_sample=False,
            # temperature=0,
            do_sample=True,
            max_new_tokens=self.max_tokens,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        print(text_outputs)

        return text_outputs[0]
