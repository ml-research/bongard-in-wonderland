import sys

sys.path.append("/workspace")
from gemini.main import Gemini
from gpt4.prompt_llm import GPT4Prompter
from claude.main import Claude
from llava.main import LlaVaPrompter
import os
import argparse


def main(args):

    model = args.model

    if model == "gpt-4o":
        prompter = GPT4Prompter(model=model)
    elif model == "llava_1.6" or model == "llava_1.5":
        prompter = LlaVaPrompter()
    elif model == "gemini":
        prompter = Gemini()
    elif model == "claude":
        prompter = Claude()
    else:
        raise ValueError(f"Model {model} not supported")

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

        for seed in range(1, 4):

            response = prompter.prompt_with_images(
                prompt, [image_path], system_prompt=system_prompt, seed=seed
            )

            # save response to file
            response_path = (
                f"results/bongard/zero_shot/{model}/BP_{id+1}/response_seed_{seed}.txt"
            )

            if not os.path.exists(os.path.dirname(response_path)):
                os.makedirs(os.path.dirname(response_path))

            with open(response_path, "w") as file:
                file.write(response)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    main(args)
