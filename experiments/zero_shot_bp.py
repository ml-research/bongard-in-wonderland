import sys

sys.path.append("/workspace")

from models.gemini.main import Gemini
from models.gpt4.prompt_llm import GPT4Prompter
from models.claude.main import Claude
from models.llava_onevision.main import LlavaOnevisionModel
from models.qwen2_vl.main import Qwen2VL
from models.internvl2_5.main import InternVL2_5

import os
import argparse
import time
import json


def get_model_prompter(model):
    if model == "gpt-4o":
        prompter = GPT4Prompter(model=model)
    elif "o1" in model:
        prompter = GPT4Prompter(model=model)
    elif "gemini" in model:
        prompter = Gemini(model)
    elif "claude" in model:
        prompter = Claude()
    elif model == "LlavaOnevision":
        prompter = LlavaOnevisionModel()
    elif model == "Qwen2VL":
        prompter = Qwen2VL()
    elif model == "InternVL2_5":
        prompter = InternVL2_5()
    else:
        raise ValueError(f"Model {model} not supported")
    return prompter


def main(args):

    model = args.model

    prompter = get_model_prompter(model)

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

    times = {}

    for id, image_path in enumerate(image_paths):

        bp_id = id + 1

        print(f"Processing BP {id+1} ...")

        times[bp_id] = []

        for i in range(1, 4):

            start = time.time()

            response = prompter.prompt_with_images(
                prompt, [image_path], system_prompt=system_prompt, seed=i
            )

            end = time.time()
            times[bp_id].append(end - start)

            # save response to file
            response_path = (
                f"results/bongard/zero_shot/{model}/BP_{id+1}/response_run_{i}.txt"
            )

            if not os.path.exists(os.path.dirname(response_path)):
                os.makedirs(os.path.dirname(response_path))

            with open(response_path, "w") as file:
                file.write(response)

        # save times
        times_path = f"results/bongard/zero_shot/{model}/times.json"

        with open(times_path, "w") as file:
            json.dump(times, file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    main(args)
