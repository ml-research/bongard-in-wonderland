import json
import sys

from models.gpt4.prompt_llm import GPT4Prompter

spiral_promt = """
Your task is to determine the direction in which a spiral depicted in a 2D black and white diagram is turning. 
The given diagram shows a spiral-like shape. In which direction is the spiral turning, starting from the center?

Please decide carefully whether the spiral is turning in clockwise or counterclockwise direction. Take a deep breath and think step-by-step. Give your answer in the following format:
```
answer = {
    "direction": <your answer>
}
```
where <your answer> can be either "counterclockwise" or "clockwise".
"""

cavity_prompt = """
Your task is to determine the position of a circle in relation to a cavity in a 2D black and white diagram.

The given diagram shows a shape with a cavity. From inside the figure, you need to decide if the circle is on the left or the right of the cavity. Carefully analyze the diagram step-by-step to identify the correct side.

Please decide carefully. Take a deep breath and think step-by-step. Give your answer in the following format:

```python
answer = {
    "position": <your answer>
}
```
where <your answer> can be either "left" if the circle is to the left of the cavity or "right" if the circle is to the right of the cavity.
"""


def main():

    solution_path = "data/solutions/bp_solutions.json"

    folder_for_prompts = "prompts/concepts"

    with open(solution_path, "r") as file:
        solutions = json.load(file)

    prompter = GPT4Prompter(model="gpt-4o-2024-08-06")

    # iterate over dict
    for key, value in solutions.items():

        prompt = f"""

        You are given a pair of contrary concepts. Your task is to craft a single prompt that asks which of those concepts appears in an image.

        # Examples for reference:

        ## Example 1:
        Contrary Concepts: [
            "Spiral curls counterclockwise",
            "Spiral curls clockwise"
        ]
        Prompt: {spiral_promt}

        ## Example 2:
        Contrary Concepts: [
            "A circle is at the left of the cavity if you look from inside the figure",
            "A circle is at the right of the cavity if you look from inside the figure"
        ],
        Prompt: {cavity_prompt}

        ## Your task:
        Please formulate a new prompt for the contrary concepts below that asks which one is present in the given image.
        Contrary Concepts: {value}
        """

        # ask llm for prompt
        response = prompter.prompt(prompt)

        # save prompt as file
        with open(f"{folder_for_prompts}/concept_prompt_{key}.txt", "w") as file:
            file.write(response)


def collect_contrary_concepts():

    path_to_prompts = "prompts/concepts"

    concepts = {}

    for i in range(1, 101):
        with open(f"{path_to_prompts}/concept_prompt_{i}.txt", "r") as file:
            prompt = file.read()

        # get contrary concepts by parsing elements in ""
        contrary_concepts = [x for x in prompt.split('"') if x.strip()]

        c1 = contrary_concepts[-4]
        c2 = contrary_concepts[-2]
        print(c1, c2)

        concepts[i] = [c1, c2]

    print(concepts)

    # save concepts as json
    target_path = "prompts/concepts/contrary_concepts.json"
    with open(target_path, "w") as file:
        json.dump(concepts, file)


if __name__ == "__main__":
    # main()
    collect_contrary_concepts()
