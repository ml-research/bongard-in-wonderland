import json
import sys
import argparse
import string
import pandas as pd

sys.path.append("gpt4")
from main import GPT4Prompter


class LLMJudge:
    def __init__(self, model="gpt-4o"):
        self.model = GPT4Prompter(model=model)
        self.system_prompt = open("prompts/bongard/system_prompt.txt", "r").read()
        self.base_prompt = open("prompts/llm_judge/judge_prompt.txt", "r").read()

    def judge_answer(self, left_rule, right_rule, solution):
        current_judge_prompt = (
            self.base_prompt.replace("LEFT_RULE_SOLUTION", solution[0])
            .replace("RIGHT_RULE_SOLUTION", solution[1])
            .replace("LEFT_RULE_ANSWER", left_rule)
            .replace("RIGHT_RULE_ANSWER", right_rule)
        )

        response = self.model.prompt(
            current_judge_prompt, system_prompt=self.system_prompt, temp=0.0
        )
        response_content = parse_answer(response)
        return response_content["answer"]

    def judge_left_answer(self, rule, solution):
        current_judge_prompt = self.base_prompt.replace(
            "RULE_SOLUTION", solution
        ).replace("RULE_ANSWER", rule)

        response = self.model.prompt(
            current_judge_prompt, system_prompt=self.system_prompt, temp=0.0
        )
        response_content = parse_answer(response)
        return response_content["answer"]


def parse_dict(response_content):
    try:
        # parse dict from response_content
        response_content = "{" + response_content.split("{")[1].split("}")[0] + "}"
    except:
        # raise ValueError(
        #     f"Could not parse dict from response_content: {response_content}"
        # )
        return {
            "set A rule": "",
            "set B rule": "",
        }

    # change ' to " for json parsing
    response_content = response_content.replace("\n", "")
    response_content = response_content.replace('"', "'")
    response_content = response_content.replace(" '", ' "')
    response_content = response_content.replace("' ", '" ')
    response_content = response_content.replace("',", '",')
    response_content = response_content.replace("':", '":')
    response_content = response_content.replace("'}", '"}')
    response_content = response_content.replace('"+"', "'+'")
    response_content = response_content.replace('"Б"', "'Б'")
    response_content = response_content.replace('"A"', "'A'")
    # same with small letters
    response_content = response_content.replace('"a"', "'a'")
    response_content = response_content.replace('"B"', "'B'")
    # v
    response_content = response_content.replace('"v"', "'v'")
    # c
    response_content = response_content.replace('"c"', "'c'")

    response_content = response_content.replace('"Б', "'Б")
    response_content = response_content.replace("\"A'", "'A'")
    # same with small letters
    response_content = response_content.replace('"a', "'a")
    response_content = response_content.replace("\"B'", "'B'")

    response_content = response_content.replace('"Y"', "'Y'")
    response_content = response_content.replace("\"Y'", "'Y'")
    response_content = response_content.replace('"x"', "'x'")
    response_content = response_content.replace("\"x'", "'x'")
    # same with o
    response_content = response_content.replace('"o"', "'o'")
    response_content = response_content.replace("\"o'", "'o'")
    response_content = response_content.replace('"L"', "'L'")
    response_content = response_content.replace("\"L'", "'L'")
    # same with B
    response_content = response_content.replace('"B"', "'B'")
    response_content = response_content.replace("\"B'", "'B'")
    # D
    response_content = response_content.replace('"D"', "'D'")
    response_content = response_content.replace("\"D'", "'D'")
    # P
    response_content = response_content.replace('"P"', "'P'")
    response_content = response_content.replace("\"P'", "'P'")

    response_content = response_content.replace("\\", "")
    response_content = response_content.replace("shape's", "shapes")
    response_content = response_content.replace(",}", "}")

    try:
        response_content = json.loads(response_content)
    except:
        print(f"Could not parse json from response_content: {response_content}")
        return {"set A rule": "", "set B rule": ""}

    return response_content


def parse_answer(response_content):
    try:
        # parse dict from response_content
        response_content = "{" + response_content.split("{")[1].split("}")[0] + "}"
    except:
        raise ValueError(
            f"Could not parse dict from response_content: {response_content}"
        )

    # change ' to " for json parsing

    response_content = response_content.replace(" '", ' "')
    response_content = response_content.replace("' ", '" ')
    response_content = response_content.replace('n"t', "n't")
    response_content = response_content.replace("\n", "")

    response_content = response_content.replace('0"', "0")
    response_content = response_content.replace('1"', "1")

    # letters
    all_small_letters = list(string.ascii_lowercase)
    all_capital_letters = list(string.ascii_uppercase)

    for letter in all_small_letters:
        response_content = response_content.replace(f'"{letter}"', f"'{letter}'")
        response_content = response_content.replace(f"{letter}'s", f"{letter}s")

    for letter in all_capital_letters:
        response_content = response_content.replace(f'"{letter}"', f"'{letter}'")
        response_content = response_content.replace(f"{letter}'s", f"{letter}s")

    response_content = response_content.replace('"Б"', "'Б'")

    try:
        response_content = json.loads(response_content)
    except:
        print("Could not parse json from response_content")
        raise ValueError(
            f"Could not parse json from response_content: {response_content}"
        )

    return response_content


def get_current_judge_prompt(judge_prompt, solution, left_rule, right_rule):
    return (
        judge_prompt.replace("LEFT_RULE_SOLUTION", solution[0])
        .replace("RIGHT_RULE_SOLUTION", solution[1])
        .replace("LEFT_RULE_ANSWER", left_rule)
        .replace("RIGHT_RULE_ANSWER", right_rule)
    )


def get_current_judge_prompt_single(judge_prompt, solution, rule):
    return judge_prompt.replace("RULE_SOLUTION", solution).replace("RULE_ANSWER", rule)


def evaluate(model, mode="zero_shot"):

    scores = {}

    solutions = json.load(open("data/solutions/bp_solutions.json"))

    if mode == "zero_shot":
        path = f"results/bongard/zero_shot/{model}"
        runs = [1, 2, 3]

    for i in range(1, 101):

        scores[i] = []
        for run in runs:
            response_path = f"{path}/BP_{i}/response_run_{run}.txt"

            try:
                with open(response_path, "r") as file:
                    response = file.read()
            except:
                print(f"Could not read response from {response_path}")
                continue

            # parse answer from response
            response_content = parse_dict(response)

            # get answer
            try:
                left_rule = response_content["set A rule"]
                right_rule = response_content["set B rule"]
            except:
                if len(response_content) >= 2:
                    # get first entry of dict
                    left_rule = response_content[list(response_content.keys())[0]]
                    right_rule = response_content[list(response_content.keys())[1]]
                else:
                    raise ValueError(
                        f"Could not parse rules from response_content: {response_content}"
                    )

            print(f"BP {i} Run {run}: {left_rule} vs {right_rule}")

            # use LLMJudge
            judge = LLMJudge()

            score = judge.judge_answer(left_rule, right_rule, solutions[str(i)])

            scores[i].append(score)

    # remove empty entries
    scores = {k: v for k, v in scores.items() if len(v) > 0}

    # save scores
    scores_path = f"{path}/scores.json"
    with open(scores_path, "w") as file:
        json.dump(scores, file)


def aggregate_answers(model):

    # create dataframe with bp_id as index and three models answers as columns
    df = pd.DataFrame(
        index=[str(x) for x in range(1, 100)],
        columns=[
            "trial 1 answer left",
            "trial 1 answer right",
            "judge 1",
            "trial 2 answer left",
            "trial 2 answer right",
            "judge 2",
            "trial 3 answer left",
            "trial 3 answer right",
            "judge 3",
        ],
    )

    path = f"results/bongard/zero_shot/{model}"

    for i in range(1, 101):

        scores = json.load(open(f"{path}/scores.json"))

        for run in [1, 2, 3]:

            bp_scores = scores[str(i)][run - 1]

            # parse model response
            response_path = f"{path}/BP_{i}/response_run_{run}.txt"

            # read response from file
            with open(response_path, "r") as file:
                response = file.read()

            response = parse_dict(response)

            df.loc[str(i), f"trial {run} answer left"] = response["set A rule"]
            df.loc[str(i), f"trial {run} answer right"] = response["set B rule"]

            df.loc[str(i), f"judge {run}"] = bp_scores

    # save df as csv
    df.to_csv(f"{path}/aggregated_scores.csv", sep=",", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--mode", type=str, default="zero_shot")

    args = parser.parse_args()

    models = [
        "o1",
        "gpt-4o",
        "claude-3-5-sonnet-20241022",
        "gemini-2.0-flash-exp",
        "gemini",
        "LlavaOnevision",
        "Qwen2VL",
        "InternVL2_5",
    ]

    for model in models:

        args.model = model
        evaluate(args.model, mode=args.mode)

        aggregate_answers(args.model)
