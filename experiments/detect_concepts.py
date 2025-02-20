import sys
import argparse

sys.path.append("/workspace")

from experiments.zero_shot_bp import get_model_prompter

from models.gemini.main import Gemini
from models.gpt4.prompt_llm import GPT4Prompter
from models.claude.main import Claude
from models.llava_onevision.main import LlavaOnevisionModel
from models.qwen2_vl.main import Qwen2VL
from models.internvl2_5.main import InternVL2_5

import os
import time


def main(model):

    prompter = get_model_prompter(model)

    for bp_id in range(1, 101):

        print(f"Processing BP {bp_id} ...")

        # prompt_path = f"prompts/cavity/cavity.txt"
        prompt_path = f"prompts/concepts/concept_prompt_{bp_id}.txt"
        prompt = open(prompt_path, "r").read()

        i_str = str(bp_id).zfill(4)
        top_path = f"data/bongard-problems-high-res/p{i_str}"

        image_paths = [
            top_path + "/0.png",
            top_path + "/1.png",
            top_path + "/2.png",
            top_path + "/3.png",
            top_path + "/4.png",
            top_path + "/5.png",
            top_path + "/6.png",
            top_path + "/7.png",
            top_path + "/8.png",
            top_path + "/9.png",
            top_path + "/10.png",
            top_path + "/11.png",
        ]

        for run in [1, 2, 3]:

            print(f"Run {run} ...")

            for i in range(len(image_paths)):

                image_path = image_paths[i]

                system_prompt = "You are a helpful assistant that can describe images provided by the user in extreme detail. You are able to recognize abstract concepts in images like humans do. You are helping a scientist discover relevant patterns in images."

                response = prompter.prompt_with_images(
                    prompt, [image_path], system_prompt=system_prompt, seed=run
                )

                # save response to file
                response_path = (
                    f"results/concepts/{model}/BP_{bp_id}/{run}/response_img_{i}.txt"
                )

                if not os.path.exists(os.path.dirname(response_path)):
                    os.makedirs(os.path.dirname(response_path))

                with open(response_path, "w") as file:
                    file.write(response)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    main(args.model)
