import sys

sys.path.append("/workspace")

from experiments.zero_shot_bp import get_model_prompter
from models.gemini.main import Gemini
from models.gpt4.prompt_llm import GPT4Prompter
from models.claude.main import Claude
from models.llava_onevision.main import LlavaOnevisionModel
from models.qwen2_vl.main import Qwen2VL
from models.internvl2_5.main import InternVL2_5

import os
import argparse


def main(args):

    model = args.model
    mode = args.mode

    prompter = get_model_prompter(model)

    system_prompt_path = "prompts/bongard/system_prompt.txt"
    system_prompt = open(system_prompt_path, "r").read()

    if mode == "left_and_right":
        prompt_path = "prompts/hypotheses/hypotheses_prompt_left_and_right.txt"
    else:
        raise ValueError(f"Mode {mode} not supported")

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

        for i in range(1, 2):

            response = prompter.prompt_with_images(
                prompt, [image_path], system_prompt=system_prompt, seed=i
            )

            # save response to file
            response_path = (
                f"results/hypotheses/{mode}/{model}/BP_{id+1}/response_run_{i}.txt"
            )

            if not os.path.exists(os.path.dirname(response_path)):
                os.makedirs(os.path.dirname(response_path))

            with open(response_path, "w") as file:
                file.write(response)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--mode", type=str, default="left_and_right")
    args = parser.parse_args()

    main(args)
