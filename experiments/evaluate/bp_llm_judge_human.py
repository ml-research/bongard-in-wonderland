import json
import sys
import argparse
import os
import numpy as np
import pandas as pd


from bp_llm_judge import LLMJudge


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
    response_content = response_content.replace("{'", '{"')
    response_content = response_content.replace('"+"', "'+'")
    response_content = response_content.replace('"Б"', "'Б'")
    # response_content = response_content.replace('"A"', "'A'")
    # same with small letters
    #  response_content = response_content.replace('"a"', "'a'")
    response_content = response_content.replace('"B"', "'B'")

    response_content = response_content.replace('"Б', "'Б")
    response_content = response_content.replace("\"A'", "'A'")
    # same with small letters
    # response_content = response_content.replace('"a', "'a")
    response_content = response_content.replace("\"B'", "'B'")

    response_content = response_content.replace('"Y"', "'Y'")
    response_content = response_content.replace("\"Y'", "'Y'")
    response_content = response_content.replace('"x"', "'x'")
    response_content = response_content.replace("\"x'", "'x'")
    # same with o
    # response_content = response_content.replace('"o"', "'o'")
    # response_content = response_content.replace("\"o'", "'o'")
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

    # handle 'A's and 'B's
    response_content = response_content.replace("'A'", "A")
    response_content = response_content.replace("'B'", "B")

    # handle A's and B's
    response_content = response_content.replace("A'", "A")
    response_content = response_content.replace("B'", "B")

    # handle " cross"
    response_content = response_content.replace('" cross"', "cross")

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


def evaluate(folder, start=2):

    scores = {}

    solutions = json.load(open("data/solutions/bp_solutions.json"))

    system_prompt_path = "prompts/bongard/system_prompt.txt"
    system_prompt = open(system_prompt_path, "r").read()

    judge_prompt_path = "prompts/llm_judge/judge_prompt.txt"
    judge_prompt = open(judge_prompt_path, "r").read()

    path = f"results/bongard/human/{folder}"

    # get all files in folder
    subject_paths = [
        f"{path}/{f}"
        for f in os.listdir(path)
        if not "scores" in f and not "complete" in f
    ]

    for subject in subject_paths:

        # read whole file
        with open(subject, "r") as file:
            content = file.read()

        # read file as pandas dataframe
        df = pd.read_csv(subject, sep="\t", header=None)
        # name first column "set A answer"
        df.columns = ["json answer"]

        df["bp id"] = [i + start for i in range(0, len(df))]

        df["set A answer"] = df["json answer"].apply(
            lambda x: json.loads(x)["set A rule"]
        )
        df["set B answer"] = df["json answer"].apply(
            lambda x: json.loads(x)["set B rule"]
        )

        df["set A solution"] = [
            x[0] for x in list(solutions.values())[start - 1 : len(df) + (start - 1)]
        ]
        df["set B solution"] = [
            x[1] for x in list(solutions.values())[start - 1 : len(df) + (start - 1)]
        ]

        df["llm judge judgement"] = 0

        # split content by \n
        content = content.split("\n")

        for i in range(0, 99):

            bp_id = i + start

            scores[bp_id] = []

            if i > len(content) - 1:
                continue

            response = content[i]

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

            print(f"BP {bp_id} {subject}: {left_rule} vs {right_rule}")

            if left_rule == "" or right_rule == "":
                score = 0
            # check if rule is nan
            elif str(left_rule) == "nan" or str(right_rule) == "nan":
                score = 0

            else:
                # get current judge prompt
                current_solution = solutions[str(bp_id)]
                current_judge_prompt = get_current_judge_prompt(
                    judge_prompt, current_solution, left_rule, right_rule
                )

                # prompt with current judge prompt
                judge = LLMJudge()
                score = judge.judge_answer(left_rule, right_rule, current_solution)

            # get score
            print(f"Score: {score}")
            scores[bp_id].append(score)

            # save score in dataframe
            df.loc[i, "llm judge judgement"] = score

        # remove empty entries
        scores = {k: v for k, v in scores.items() if len(v) > 0}

        # save scores
        scores_path = f"{subject}_scores.json"
        with open(scores_path, "w") as file:
            json.dump(scores, file)

        # save dataframe
        df_path = f"{subject}_scores.csv"
        df.to_csv(df_path, sep=",", index=False)

        # print scores
        print(f"Subject {subject}: {scores}")
        # count 1s
        count = 0
        for key in scores.keys():
            count += np.sum(scores[key])
        print(f"Subject {subject}: {count} correct answers")


def aggregate_human_scores(folder):

    path = f"results/bongard/human/{folder}"

    # get all files in folder
    subjects = [
        f for f in os.listdir(path) if "scores.json" in f and not "complete" in f
    ]

    df = pd.DataFrame(index=[str(x) for x in range(2, 100)], columns=subjects)

    scores = {}

    for subject in subjects:

        # open scores
        scores_path = f"{path}/{subject}"
        scores = json.load(open(scores_path))

        for key, value in scores.items():
            df.loc[key, subject] = value[0]

    # remove "_scores.json" from subject names
    df.columns = [x.replace("_scores.json", "") for x in subjects]

    # save df as csv
    df.to_csv(f"{path}/complete_scores.csv", sep=",", index=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="gpt-4o")

    args = parser.parse_args()

    evaluate(args.folder, start=2)
    aggregate_human_scores(args.folder)
