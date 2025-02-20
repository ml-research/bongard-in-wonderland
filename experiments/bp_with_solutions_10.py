import random
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
import json
import time


def main(args):

    model = args.model

    prompter = get_model_prompter(model)

    system_prompt_path = "prompts/bongard/system_prompt.txt"
    system_prompt = open(system_prompt_path, "r").read()

    prompt_path = "prompts/bongard/with_solutions.txt"
    prompt = open(prompt_path, "r").read()

    solutions_path = "data/solutions/bp_solutions.json"
    solutions = json.load(open(solutions_path))

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

        for run in [1, 2, 3]:

            # correct solution
            current_solutions = {str(id + 1): solutions[str(id + 1)]}

            selected_ids = [id + 1]
            # pick 9 random solutions
            for i in range(9):
                random_id = id + 1
                while random_id in selected_ids:
                    random_id = random.randint(1, 100)
                # add random solution to dict
                current_solutions[str(random_id)] = solutions[str(random_id)]
                selected_ids.append(random_id)

            # sort solutions
            current_solutions = dict(sorted(current_solutions.items()))

            # replace <SOLUTIONS> with solutions
            current_prompt = prompt.replace(
                "<SOLUTIONS>", json.dumps(current_solutions)
            )

            start_time = time.time()

            response = prompter.prompt_with_images(
                current_prompt, [image_path], system_prompt=system_prompt, seed=run
            )

            end_time = time.time()
            print(f"Time taken: {end_time-start_time}")

            # save response to file
            response_path = f"results/bongard/with_solutions_10/{model}/BP_{id+1}/response_run_{run}.txt"

            if not os.path.exists(os.path.dirname(response_path)):
                os.makedirs(os.path.dirname(response_path))

            with open(response_path, "w") as file:
                file.write(response)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="o1")

    args = parser.parse_args()

    main(args)
