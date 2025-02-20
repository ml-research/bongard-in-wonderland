import torch

from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.vl import load_image

import sys

print(sys.path)

from rtpt import RTPT


class InternVL2_5:
    def __init__(self):

        self.rtpt = RTPT(
            name_initials="XX", experiment_name="InternVL2.5", max_iterations=10
        )
        self.rtpt.start()

        self.model_name = "OpenGVLab/InternVL2_5-78B"
        number_of_devices = torch.cuda.device_count()
        d_ts = torch.zeros(1).to(f"cuda")

        engine = TurbomindEngineConfig(
            # session_len=8192,
            tp=number_of_devices,
        )

        self.model = pipeline(self.model_name, backend_config=engine)
        self.max_tokens = 2048

    def prompt_with_images(
        self, prompt_text, image_paths, system_prompt=None, seed=None
    ):

        image_path = image_paths[0]
        image = load_image(image_path)

        hyperparameters = {
            "do_sample": True,
            "max_new_tokens": self.max_tokens,
        }
        gen_config = GenerationConfig(**hyperparameters)

        response = self.model((prompt_text, image), gen_config=gen_config)
        print(response)

        return response.text


def main():
    import os

    prompter = InternVL2_5()

    system_prompt_path = "prompts/bongard/system_prompt.txt"
    system_prompt = open(system_prompt_path, "r").read()

    prompt_path = "prompts/bongard/zero_shot.txt"
    prompt = open(prompt_path, "r").read()

    top_path = "data/bpimgs"

    # get all files in folder
    image_paths = [
        os.path.join(top_path, f)
        for f in os.listdir(top_path)
        if os.path.isfile(os.path.join(top_path, f))
    ]
    # sort files
    image_paths.sort()

    for id, image_path in enumerate(image_paths):

        print(f"Processing BP {id+1} ...")

        for run in range(1, 4):

            response = prompter.prompt_with_images(prompt, [image_path])

            # save response to file
            response_path = f"results/bongard/zero_shot/InternVL2_5-78B/BP_{id+1}/response_run_{run}.txt"

            if not os.path.exists(os.path.dirname(response_path)):
                os.makedirs(os.path.dirname(response_path))

            with open(response_path, "w") as file:
                file.write(response)


if __name__ == "__main__":

    main()
