from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from rtpt import RTPT


class Qwen2VL:
    def __init__(self):

        self.rtpt = RTPT(
            name_initials="XX", experiment_name="Qwen2VL", max_iterations=10
        )
        self.rtpt.start()

        print("Loading Qwen2-VL-72B-Instruct model...")
        # default: Load the model on the available device(s)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
        )
        print("Model loaded.")
        # default processer

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-72B-Instruct", min_pixels=431 * 431, max_pixels=2234 * 1409
        )
        print("Tokenizer loaded.")
        self.max_tokens = 2048
        self.model_name = "Qwen2-VL-72B-Instruct"

    def prompt_with_images(
        self, prompt_text, image_paths, system_prompt=None, seed=None
    ):

        # TODO: url or path?
        # images = [Image.open(path) for path in image_paths]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_paths[0]},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Preprocess the inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print("text:", text)

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            # do_sample=False,
            do_sample=True,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        print(output_text)

        return output_text[0]
