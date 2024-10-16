import anthropic
import base64


def get_base64_encoded_image(image_path):

    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode("utf-8")
        return base64_string


class Claude:
    def __init__(self):

        # load api key from file
        api_key = open("claude/api_key.txt", "r").read()
        self.model = anthropic.Anthropic(api_key=api_key)
        self.model_name = "claude-3-5-sonnet-20240620"

    def prompt(self, prompt_text):
        response = self.model.generate_content(prompt_text)
        return response

    def prompt_with_images(
        self, prompt_text, image_paths, system_prompt=None, seed=None
    ):

        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image(image_path),
                },
            }
            for image_path in image_paths
        ]
        content += [{"type": "text", "text": prompt_text}]

        message_list = [
            {
                "role": "user",
                "content": content,
                # [
                #     {
                #         "type": "image",
                #         "source": {
                #             "type": "base64",
                #             "media_type": "image/png",
                #             "data": get_base64_encoded_image(image_paths[0]),
                #         },
                #     },
                #     {"type": "text", "text": prompt_text},
                # ],
            },
        ]

        response = self.model.messages.create(
            model=self.model_name, max_tokens=2048, messages=message_list
        )

        return response.content[0].text
