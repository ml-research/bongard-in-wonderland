import json
import numpy as np
import pandas as pd

import sys

from experiments.evaluate.model_names import map_model_names


def main(models, mode):

    # create dataframe with bp_id as index and model as column
    df = pd.DataFrame(index=[str(x) for x in range(1, 100)], columns=models)
    df_percentage = pd.DataFrame(index=[str(x) for x in range(1, 100)], columns=models)

    df_runs = pd.DataFrame(index=[str(x) for x in range(1, 100)])

    for model in models:

        if mode == "zero_shot":
            path = f"results/bongard/zero_shot/{model}/scores.json"
        elif mode == 100:
            path = f"results/bongard/with_solutions/{model}/scores.json"
        elif mode == 10:
            path = f"results/bongard/with_solutions_10/{model}/scores.json"
        elif mode == "attributes":
            path = f"results/bongard/with_attributes/{model}/scores.json"

        n_solved_bps = [0, 0, 0]

        scores = json.load(open(path))

        for bp_id, bp_scores in scores.items():

            num_solved = sum(bp_scores)
            if num_solved == 0:
                df.loc[bp_id, model] = "0/3"
            elif num_solved == 1:
                df.loc[bp_id, model] = "\cellcolor{YellowGreen!25}1/3"
            elif num_solved == 2:
                df.loc[bp_id, model] = "\cellcolor{Green!25}2/3"
            elif num_solved == 3:
                df.loc[bp_id, model] = "\cellcolor{PineGreen!25}3/3"

            df_percentage.loc[bp_id, model] = num_solved / 3

            for run, score in enumerate(bp_scores):

                if score == 1:
                    n_solved_bps[run] += 1

                df_runs.loc[bp_id, f"{model}_run_{run+1}"] = score

        print(f"Model: {model}")
        print(f"Run 1: {n_solved_bps[0]}")
        print(f"Run 2: {n_solved_bps[1]}")
        print(f"Run 3: {n_solved_bps[2]}")

        # get mean
        mean_solved_bps = sum(n_solved_bps) / len(n_solved_bps)
        # get std
        n_solved_bps_std = np.std(n_solved_bps)

        print(f"Mean: {mean_solved_bps}")
        print(f"Std: {n_solved_bps_std}")

    # rename columns
    df.columns = [map_model_names(model) for model in df.columns]
    # print df as latex table
    print(df.to_latex())

    # save df as csv
    df_percentage.to_csv(f"results/bongard/summary_{mode}.csv")
    df_runs.to_csv(f"results/bongard/summary_runs_{mode}.csv")


if __name__ == "__main__":

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

    main(models, mode="zero_shot")
    main(models, mode=10)
    main(models, mode=100)
