import json
import sys
import argparse

sys.path.append("gpt4")
from prompt_llm import GPT4Prompter


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
    response_content = response_content.replace('"B"', "'B'")

    response_content = response_content.replace('"Б', "'Б")
    response_content = response_content.replace("\"A'", "'A'")
    response_content = response_content.replace("\"B'", "'B'")

    response_content = response_content.replace('"Y"', "'Y'")
    response_content = response_content.replace("\"Y'", "'Y'")
    response_content = response_content.replace('"x"', "'x'")
    response_content = response_content.replace("\"x'", "'x'")
    response_content = response_content.replace('"L"', "'L'")
    response_content = response_content.replace("\"L'", "'L'")

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

    try:
        response_content = json.loads(response_content)
    except:
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

    system_prompt_path = "prompts/bongard/system_prompt.txt"
    system_prompt = open(system_prompt_path, "r").read()

    judge_prompt_path = "prompts/llm_judge/judge_prompt.txt"
    judge_prompt = open(judge_prompt_path, "r").read()

    if mode == "zero_shot":
        path = f"results/bongard/zero_shot/{model}"
        seeds = [1, 2, 3]
    elif mode == "zero_shot_human":
        path = f"results/bongard/zero_shot_human/{model}"
        seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    elif mode == "attributes":
        path = f"results/bongard/with_attributes/{model}"
        seeds = [1, 2, 3]

    for i in range(1, 101):

        scores[i] = []
        for seed in seeds:
            response_path = f"{path}/BP_{i}/response_seed_{seed}.txt"

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

            print(f"BP {i} Seed {seed}: {left_rule} vs {right_rule}")

            # get current judge prompt
            current_solution = solutions[str(i)]
            current_judge_prompt = get_current_judge_prompt(
                judge_prompt, current_solution, left_rule, right_rule
            )

            # prompt with current judge prompt
            prompter = GPT4Prompter(model="gpt-4o")
            response = prompter.prompt(
                current_judge_prompt, system_prompt=system_prompt
            )

            # parse answer from response
            try:
                response_content = parse_answer(response)
            except:
                print(f"Could not parse answer from response: {response}")
                response_content = {"answer": -1}
                continue

            # get score
            score = response_content["answer"]
            scores[i].append(score)

    # remove empty entries
    scores = {k: v for k, v in scores.items() if len(v) > 0}

    # save scores
    scores_path = f"{path}/scores.json"
    with open(scores_path, "w") as file:
        json.dump(scores, file)


def evaluate_single(model, mode="zero_shot"):
    scores = {}

    solutions = json.load(open("data/solutions/bp_solutions.json"))

    system_prompt_path = "prompts/bongard/system_prompt.txt"
    system_prompt = open(system_prompt_path, "r").read()

    judge_prompt_path = "prompts/llm_judge/judge_prompt_single.txt"
    judge_prompt = open(judge_prompt_path, "r").read()

    if mode == "zero_shot":
        path = f"results/bongard/zero_shot/{model}"
    elif mode == "attributes":
        path = f"results/bongard/with_attributes/{model}"

    for i in range(1, 101):

        scores[i] = []
        for seed in [1, 2, 3]:
            response_path = f"{path}/BP_{i}/response_seed_{seed}.txt"

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

            print(f"BP {i} Seed {seed}: {left_rule} vs {right_rule}")

            bp_scores = [0, 0]

            for id, rule in enumerate([left_rule, right_rule]):

                # get current judge prompt
                current_solution = solutions[str(i)]
                current_judge_prompt = get_current_judge_prompt_single(
                    judge_prompt, current_solution[id], rule
                )

                # prompt with current judge prompt
                prompter = GPT4Prompter(model="gpt-4o")
                response = prompter.prompt(
                    current_judge_prompt, system_prompt=system_prompt
                )

                # parse answer from response
                try:
                    response_content = parse_answer(response)
                except:
                    print(f"Could not parse answer from response: {response}")
                    response_content = {"answer": -1}
                    continue

                # get score
                bp_scores[id] = response_content["answer"]

            scores[i].append(bp_scores)

    # save scores
    scores_path = f"{path}/scores_single.json"
    with open(scores_path, "w") as file:
        json.dump(scores, file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--mode", type=str, default="zero_shot")

    args = parser.parse_args()

    evaluate(args.model, mode=args.mode)
