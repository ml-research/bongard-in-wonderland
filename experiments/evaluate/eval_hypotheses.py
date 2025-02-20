import json
import sys
import pandas as pd

sys.path.append("/workspace")

from experiments.evaluate.bp_llm_judge import LLMJudge
from experiments.evaluate.model_names import map_model_names


def parse_list(path):
    try:
        # read file line by line
        with open(path, "r") as file:
            response_content = file.read()

        # split via "\n"
        response_content = response_content.split("\n")

        hypotheses = []
        start_read = False
        pair = []

        for element in response_content:
            if "```" in element:
                start_read = not start_read
                continue
            if start_read:
                if "[" in element or "]" in element:
                    continue
                if element == "":
                    continue
                if not '"' in element:
                    continue
                if element.strip() == "(" or element.strip() == ")":
                    continue
                if (
                    "(" in element
                    and ")" in element
                    and not ('"' in element.split(")")[-1])
                ):
                    # element = element.split("(")[1].split(")")[0]
                    element = element.split('("')[1:]
                    element = "".join(element)
                    element = element.split('")')[:-1]
                    element = "".join(element)

                    element = element.split('",')
                    element[0] = element[0]

                    if len(element) < 2:
                        continue

                    pair = [element[0].strip(), element[1].strip()]
                    # remove " and ' from string
                    pair = [x.replace('"', "").replace("'", "") for x in pair]
                    hypotheses.append(pair)

                else:
                    element = (
                        element.strip()
                        .replace('"', "")
                        .replace("'", "")
                        .replace("(", "")
                        .replace(")", "")
                        .replace(",", "")
                    )
                    pair.append(element.strip())
                    if len(pair) == 2:
                        hypotheses.append(pair)
                        pair = []

        return hypotheses

    except:
        # raise ValueError(
        #     f"Could not parse dict from response_content: {response_content}"
        # )
        print(f"Could not read response from {path}")
        return []


def main(args):
    """Function that evaluates the model responses to the 100 Bongard problems. Each answer has 20 hypotheses that need to be evaluated against the ground truth solutions."""

    model = args.model
    mode = args.mode

    solutions = json.load(open("data/solutions/bp_solutions.json"))

    llm_judge = LLMJudge()

    # open correct hypotheses (if exists)
    try:
        with open(
            f"results/hypotheses/{mode}/{model}/correct_hypotheses.json", "r"
        ) as file:
            correct_hypotheses = json.load(file)
    except:
        correct_hypotheses = {}

    for bongard_id in range(1, 101):

        if str(bongard_id) in correct_hypotheses.keys():
            if len(correct_hypotheses[str(bongard_id)]) > 0:
                continue
        else:
            correct_hypotheses[str(bongard_id)] = []

        hypotheses_path = (
            f"results/hypotheses/{mode}/{model}/BP_{bongard_id}/response_run_1.txt"
        )

        # parse hypotheses as json
        hypotheses = parse_list(hypotheses_path)

        # get correct solution
        correct_solution = solutions[str(bongard_id)]

        print(f"BP {bongard_id} ...")

        for hypothesis in hypotheses:

            response = llm_judge.judge_answer(
                hypothesis[0], hypothesis[1], correct_solution
            )

            correct_hypotheses[str(bongard_id)].append(response)

            if response == 1:
                print(f"Correct hypothesis: {hypothesis}")
                break

    # save correct hypotheses
    with open(
        f"results/hypotheses/{mode}/{model}/correct_hypotheses.json", "w"
    ) as file:
        json.dump(correct_hypotheses, file)


def collect_results(model):
    """Function that collects the results from the evaluation of the hypotheses."""

    number_correct_hypotheses = 0
    for mode in ["left_and_right"]:

        try:
            with open(
                f"results/hypotheses/{mode}/{model}/correct_hypotheses.json",
                "r",
            ) as file:
                correct_hypotheses = json.load(file)
        except:
            print(f"Could not read correct hypotheses for {model} in mode {mode}")
            correct_hypotheses = {}

    # iterate over all values and count if sum is greater 1
    for key, value in correct_hypotheses.items():
        number_correct_hypotheses += sum(value)

    print(f"Model {model} has {number_correct_hypotheses} correct hypotheses.")
    return number_correct_hypotheses


def aggregate_results(models, mode="left_and_right"):

    df = pd.DataFrame(index=[str(x) for x in range(1, 101)], columns=models)

    for model in models:
        # open correct hypotheses (if exists)
        try:
            with open(
                f"results/hypotheses/{mode}/{model}/correct_hypotheses.json",
                "r",
            ) as file:
                correct_hypotheses = json.load(file)
        except:
            correct_hypotheses = {}

        for key, value in correct_hypotheses.items():
            df.loc[key, model] = 1 if sum(value) > 0 else 0

    df.to_csv("results/hypotheses/all_results.csv", sep=",", index=True)

    # for each cell, if 1, color green, if 0, no color
    df_latex = df.copy()

    # map column names to other name
    df_latex.columns = [map_model_names(x) for x in df_latex.columns]

    for column in df_latex.columns:
        df_latex[column] = df_latex[column].apply(
            lambda x: f"\\cellcolor{{Green!25}}1" if x == 1 else "0"
        )

    print(df_latex.to_latex())


def eval_subsets():

    df_of_solved_bps = pd.read_csv(
        "results/for_plotting/single_bp_results_humans_and_models.csv", index_col=0
    )
    df_of_concepts = pd.read_csv("results/concepts/summary.csv", index_col=0)
    df_of_hypotheses = pd.read_csv("results/hypotheses/all_results.csv", index_col=0)

    # for each model, get bps with correct hypothesis and solved
    for model in df_of_hypotheses.columns:

        solved_bps_q1 = [
            idx + 1
            for idx, value in enumerate(df_of_solved_bps[model].values)
            if value >= 30
        ]

        solved_bps_q3 = [
            idx + 1
            for idx, value in enumerate(df_of_concepts[model].values)
            if value >= 0.8
        ]

        solved_bps_q4 = [
            idx + 1
            for idx, value in enumerate(df_of_hypotheses[model].values)
            if value == 1
        ]

        # get intersection
        solved_bps_q4_and_q1 = list(set(solved_bps_q1).intersection(set(solved_bps_q4)))

        # print(f"Model {model} solved {len(solved_bps_q4_and_q1)} BPs in Q1 and Q4")

        # solved in q4 but not q1
        solved_bps_q4_not_q1 = list(set(solved_bps_q4) - set(solved_bps_q1))

        # print(
        #     f"Model {model} solved {len(solved_bps_q4_not_q1)} BPs in Q4 but not in Q1"
        # )

        solved_q4_and_q3_not_q1 = set(solved_bps_q4).intersection(
            set(solved_bps_q3)
        ) - set(solved_bps_q1)

        print(
            f"Model {model} solved {len(solved_q4_and_q3_not_q1)} BPs in Q4 and Q3 but not in Q1"
        )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Evaluate hypotheses")

    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        default="gpt-4o",
        help="Model to evaluate hypotheses for",
    )

    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="left_and_right",
        help="Mode to evaluate hypotheses in",
    )

    args = parser.parse_args()

    collect_results(args.model)

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
    aggregate_results(models)
    eval_subsets()
