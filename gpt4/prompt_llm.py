import base64
import os
import requests
from openai import OpenAI
from datetime import datetime


class GPT4Prompter:
    def __init__(self, model="gpt-4o", seed=42):
        # load the API key from "open-ai-key"
        with open("gpt4/open-ai-key", "r") as file:
            api_key = file.read().strip()

        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        if model == "gpt-4o":
            model = "gpt-4o-2024-08-06"
        self.model = model
        self.seed = seed
        self.system_fingerprint = None

    def prompt(self, prompt_text, system_prompt=None, seed=None):
        """Generate a response to a prompt using the OpenAI API."""

        if seed is None:
            seed = self.seed

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
            seed=seed,
            max_tokens=2000,
        )
        # Get system fingerprint
        self.system_fingerprint = response.system_fingerprint
        self.model = response.model

        response_text = response.choices[0].message.content.strip()

        # self._log_prompt(prompt_text, response_text)

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
        seed=None,
        temperature=0.2,
        top_p=0.1,
    ):

        if seed is None:
            seed = self.seed

        if system_prompt is None:
            system_prompt = "You are a helpful assistant that can describe images provided by the user in extreme detail. You are able to recognize abstract concepts in images like humans do. You are helping a scientist discover relevant patterns in images."

        # encode images
        encoded_images = [self._encode_image(path) for path in paths]

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
            messages[1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        # TODO: change to png?
                        "url": f"data:image/jpeg;base64,{image}",
                    },
                }
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                seed=seed,
                top_p=top_p,
                temperature=temperature,
            )
        except Exception as e:
            print("Error: ", e)
            return ""

        response_content = response.choices[0].message.content

        return response_content

    def _log_prompt(self, prompt_text, response):
        # create response folder if it does not exist
        if not os.path.exists(f"llm/{self.bongard_id}/responses"):
            os.makedirs(f"llm/{self.bongard_id}/responses")

        # Save the response together with prompt to file
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log = prompt_text + "\n---------------\n" + response
        with open(
            f"context/{self.bongard_id}/responses/{timestamp}_{self.seed}_{self.system_fingerprint}.txt",
            "w",
        ) as file:
            file.write(log)
