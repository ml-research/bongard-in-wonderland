import base64
import os
from openai import OpenAI
from datetime import datetime


class GPT4Prompter:
    def __init__(self, model="gpt-4o", key=None):

        if key:
            self.api_key = key
        else:
            # load the API key from "open-ai-key"
            with open("gpt4/open-ai-key-bongard", "r") as file:
                api_key = file.read().strip()
            self.api_key = api_key

        self.client = OpenAI(api_key=self.api_key)
        if model == "gpt-4o":
            model = "gpt-4o-2024-08-06"
        if model == "o1":
            model = "o1-2024-12-17"

        self.model = model
        self.system_fingerprint = None

        print(f"Using model: {model}")

    def prompt(self, prompt_text, system_prompt=None, temp=None):
        """Generate a response to a prompt using the OpenAI API."""

        if system_prompt is None:
            system_prompt = "You are a helpful assistant that can describe images provided by the user in extreme detail. You are able to recognize abstract concepts in images like humans do. You are helping a scientist discover relevant patterns in images."

        # Call the completion endpoint
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                        }
                    ],
                },
                {"role": "user", "content": prompt_text},
            ],
            model=self.model,
            max_tokens=2000,
            temperature=temp,
        )
        # Get system fingerprint
        self.system_fingerprint = response.system_fingerprint
        self.model = response.model

        response_text = response.choices[0].message.content.strip()

        return response_text

    def _encode_image(self, image_path):
        # Function to encode the image
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def prompt_with_images(
        self,
        prompt_text: str,
        paths: [str],
        system_prompt=None,
    ):

        if system_prompt is None:
            system_prompt = "You are a helpful assistant that can describe images provided by the user in extreme detail. You are able to recognize abstract concepts in images like humans do. You are helping a scientist discover relevant patterns in images."

        # encode images
        encoded_images = [self._encode_image(path) for path in paths]

        # Prepare messages based on the model type
        if self.model == "gpt-4o" or self.model == "gpt-4o-2024-08-06":
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        }
                    ],
                },
            ]

            for image in encoded_images:
                messages[-1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                        },
                    }
                )

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=2048,
                )
            except Exception as e:
                print("Error: ", e)
                return ""

        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        }
                    ],
                },
            ]

            for image in encoded_images:
                messages[-1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                        },
                    }
                )

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
            except Exception as e:
                print("Error: ", e)
                return ""

        response_content = response.choices[0].message.content

        return response_content
