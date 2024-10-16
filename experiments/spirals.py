import sys

sys.path.append("/workspace")
from gemini.main import Gemini
from gpt4.prompt_llm import GPT4Prompter
from claude.main import Claude
from llava.main import LlaVaPrompter

import os


def main():

    # model = "gpt-4o"
    model = "o1-preview"
    model = "claude"
    mode = "pe"  # "cot"
    # mode = "cot"
    # mode = "pe"

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

    prompt_path = f"prompts/spirals/spiral_{mode}.txt"
    prompt = open(prompt_path, "r").read()

    top_path = "data/bongard-problems-high-res/p0016"
    # top_path = "data/bongard-problems/p016"

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

    for seed in [1, 2, 3]:
        for i in range(len(image_paths)):
            image_path = image_paths[i]

            response = prompter.prompt_with_images(prompt, [image_path])

            # save response to file
            response_path = (
                f"results/spirals/{model}/{mode}/{seed}/response_img_{i}.txt"
            )

            if not os.path.exists(os.path.dirname(response_path)):
                os.makedirs(os.path.dirname(response_path))

            with open(response_path, "w") as file:
                file.write(response)


if __name__ == "__main__":
    main()
