import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import sys

from experiments.evaluate.model_names import map_model_names

size = [2, 14, 21, 22, 34, 38]
concept = [
    24,
    26,
    3,
    4,
    5,
    6,
    7,
    9,
    10,
    11,
    12,
    13,
    15,
    17,
    18,
    19,
    25,
    30,
    32,
    33,
    43,
    50,
    76,
    82,
    92,
    95,
    96,
    97,
    98,
    100,
]
number = [23, 27, 28, 29, 31, 53, 70, 71, 85, 86, 87, 88, 89, 90, 91]
spatial = [
    8,
    16,
    20,
    35,
    36,
    37,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    51,
    52,
    54,
    55,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    72,
    73,
    74,
    75,
    78,
    79,
    81,
    83,
    84,
    93,
    94,
    99,
]
same = [56, 57, 58, 59, 60, 77, 80]


def get_categpry(bp_id):
    if bp_id in size:
        return "size"
    if bp_id in concept:
        return "concept"
    if bp_id in number:
        return "number"
    if bp_id in spatial:
        return "spatial"
    if bp_id in same:
        return "same"
    return "-"


def collect_agg_results(models, mode):

    if mode == "humans":
        models = ["humans"] + models

    # df for mean values
    mean_df = pd.DataFrame(
        index=["all", "size", "concept", "number", "spatial", "same"],
        columns=models,
    )

    std_df = pd.DataFrame(
        index=["all", "size", "concept", "number", "spatial", "same"], columns=models
    )

    mean_percentage_df = pd.DataFrame(
        index=["all", "size", "concept", "number", "spatial", "same"],
        columns=models,
    )

    std_percentage_df = pd.DataFrame(
        index=["all", "size", "concept", "number", "spatial", "same"],
        columns=models,
    )

    # create dataframe with bp_id as index and model as column
    df = pd.DataFrame(index=[str(x) for x in range(1, 100)], columns=models)

    for model in models:

        if mode == "humans":
            if model == "humans":
                path = "results/bongard/human/complete_final/scores.json"
                df_human_scores = pd.read_csv(
                    "results/bongard/human/complete_final/complete_final_aggregated.csv",
                    index_col=0,
                )
                # create dict
                scores = {}
                for i in df_human_scores.index:
                    scores[str(i + 2)] = [int(x) for x in df_human_scores.loc[i].values]

                # save scores
                with open(path, "w") as file:
                    json.dump(scores, file)

            elif model == "humans_top_5":
                path = "results/bongard/human/top_5/scores.json"
                scores = json.load(open(path))
            else:
                path = f"results/bongard/zero_shot/{model}/scores.json"
                scores = json.load(open(path))

        counter = 0
        for bp_id, bp_scores in scores.items():

            bp_percentage = round((sum(bp_scores) / len(bp_scores)) * 100, 2)
            df.loc[bp_id, model] = bp_percentage

        # add model to mean_df that contains top 5

        # existence = [24, 26]
        existence = []
        size = [2, 14, 21, 22, 34, 38]
        concept = [
            24,
            26,
            3,
            4,
            5,
            6,
            7,
            9,
            10,
            11,
            12,
            13,
            15,
            17,
            18,
            19,
            25,
            30,
            32,
            33,
            43,
            50,
            76,
            82,
            92,
            95,
            96,
            97,
            98,
            100,
        ]
        number = [23, 27, 28, 29, 31, 53, 70, 71, 85, 86, 87, 88, 89, 90, 91]
        spatial = [
            8,
            16,
            20,
            35,
            36,
            37,
            39,
            40,
            41,
            42,
            44,
            45,
            46,
            47,
            48,
            49,
            51,
            52,
            54,
            55,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            72,
            73,
            74,
            75,
            78,
            79,
            81,
            83,
            84,
            93,
            94,
            99,
        ]
        same = [56, 57, 58, 59, 60, 77, 80]

        all_bps = existence + size + concept + number + spatial + same

        # len of all bps should be 100
        assert len(all_bps) == 99

        for i in range(2, 101):
            if i not in all_bps:
                print(f"BP {i} not in all bps")

        # get mean values for existence
        # mean_df.loc["existence", model] = np.mean(
        #     [df.loc[str(bp), model] for bp in existence]
        # )

        solved_bps_all = [0] * len(scores["2"])
        solved_bps_all_percentages = [0] * len(scores["2"])
        solved_bps_size = [0] * len(scores["2"])
        solved_bps_size_percentages = [0] * len(scores["2"])
        solved_bps_concept = [0] * len(scores["2"])
        solved_bps_concept_percentages = [0] * len(scores["2"])
        solved_bps_number = [0] * len(scores["2"])
        solved_bps_number_percentages = [0] * len(scores["2"])
        solved_bps_spatial = [0] * len(scores["2"])
        solved_bps_spatial_percentages = [0] * len(scores["2"])
        solved_bps_same = [0] * len(scores["2"])
        solved_bps_same_percentages = [0] * len(scores["2"])

        for all_bp_id in all_bps:
            for run_id, score in enumerate(scores[str(all_bp_id)]):
                if score == 1:
                    solved_bps_all[run_id] += 1

        mean_df.loc["all", model] = np.mean(solved_bps_all)
        std_df.loc["all", model] = np.std(solved_bps_all)

        mean_percentage_df.loc["all", model] = np.mean(
            [n / len(all_bps) for n in solved_bps_all]
        )
        std_percentage_df.loc["all", model] = np.std(
            [n / len(all_bps) for n in solved_bps_all]
        )

        for size_bp_id in size:
            for run_id, score in enumerate(scores[str(size_bp_id)]):
                if score == 1:
                    solved_bps_size[run_id] += 1

        mean_df.loc["size", model] = np.mean(solved_bps_size)
        std_df.loc["size", model] = np.std(solved_bps_size)

        mean_percentage_df.loc["size", model] = np.mean(
            [n / len(size) for n in solved_bps_size]
        )
        std_percentage_df.loc["size", model] = np.std(
            [n / len(size) for n in solved_bps_size]
        )

        for concept_bp_id in concept:
            for run_id, score in enumerate(scores[str(concept_bp_id)]):
                if score == 1:
                    solved_bps_concept[run_id] += 1

        mean_df.loc["concept", model] = np.mean(solved_bps_concept)
        std_df.loc["concept", model] = np.std(solved_bps_concept)

        mean_percentage_df.loc["concept", model] = np.mean(
            [n / len(concept) for n in solved_bps_concept]
        )
        std_percentage_df.loc["concept", model] = np.std(
            [n / len(concept) for n in solved_bps_concept]
        )

        for number_bp_id in number:
            for run_id, score in enumerate(scores[str(number_bp_id)]):
                if score == 1:
                    solved_bps_number[run_id] += 1

        mean_df.loc["number", model] = np.mean(solved_bps_number)
        std_df.loc["number", model] = np.std(solved_bps_number)

        mean_percentage_df.loc["number", model] = np.mean(
            [n / len(number) for n in solved_bps_number]
        )
        std_percentage_df.loc["number", model] = np.std(
            [n / len(number) for n in solved_bps_number]
        )

        for spatial_bp_id in spatial:
            for run_id, score in enumerate(scores[str(spatial_bp_id)]):
                if score == 1:
                    solved_bps_spatial[run_id] += 1

        mean_df.loc["spatial", model] = np.mean(solved_bps_spatial)
        std_df.loc["spatial", model] = np.std(solved_bps_spatial)

        mean_percentage_df.loc["spatial", model] = np.mean(
            [n / len(spatial) for n in solved_bps_spatial]
        )
        std_percentage_df.loc["spatial", model] = np.std(
            [n / len(spatial) for n in solved_bps_spatial]
        )

        for same_bp_id in same:
            for run_id, score in enumerate(scores[str(same_bp_id)]):
                if score == 1:
                    solved_bps_same[run_id] += 1

        mean_df.loc["same", model] = np.mean(solved_bps_same)
        std_df.loc["same", model] = np.std(solved_bps_same)

        mean_percentage_df.loc["same", model] = np.mean(
            [n / len(same) for n in solved_bps_same]
        )
        std_percentage_df.loc["same", model] = np.std(
            [n / len(same) for n in solved_bps_same]
        )

    # map column names to other names
    mean_df.columns = [map_model_names(x) for x in mean_df.columns]
    std_df.columns = [map_model_names(x) for x in std_df.columns]
    mean_percentage_df.columns = [
        map_model_names(x) for x in mean_percentage_df.columns
    ]

    # round all values to 2 decimal places
    mean_df = mean_df.round(2)
    print(mean_df)
    std_df = std_df.round(2)

    print(std_percentage_df)

    # save df as csv
    mean_df.to_csv(f"results/for_plotting/mean_results_humans_and_models.csv")
    std_df.to_csv(f"results/for_plotting/std_results_humans_and_models.csv")

    # save df as csv
    mean_percentage_df.to_csv(
        f"results/for_plotting/mean_percentage_results_humans_and_models.csv"
    )
    std_percentage_df.to_csv(
        f"results/for_plotting/std_percentage_results_humans_and_models.csv"
    )

    # save df
    df.to_csv(f"results/for_plotting/single_bp_results_humans_and_models.csv")

    print(mean_percentage_df.to_latex(float_format="%.4f"))

    # plot mean values as bar plot with bar width 0.3
    ax = mean_df.plot(
        kind="bar", figsize=(15, 3), title="Mean values for Bongard problems", width=0.8
    )
    # set x label
    ax.set_xlabel("Bongard problem type")
    # set y label
    ax.set_ylabel("Mean % solved")
    # x ticks horizontal
    plt.xticks(rotation=0)
    # set y range from 0 to 80
    plt.ylim(0, 80)
    # place legend outside of plot
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
    # fig tight layout
    plt.tight_layout()
    # save plot
    plt.savefig("figures/human_and_vlms_values.pdf")
    plt.savefig("figures/human_and_vlms_values.png")


def main(models, mode):

    if mode == "humans":
        models.append("humans")

    # create dataframe with bp_id as index and model as column
    df = pd.DataFrame(
        index=[str(x) for x in range(1, 101)], columns=["Categories"] + models
    )

    df_ratios = pd.DataFrame(index=[str(x) for x in range(1, 101)], columns=models)

    for model in models:

        if mode == "humans":
            if model == "humans":
                path = "results/bongard/human/complete_final/scores.json"
            else:
                path = f"results/bongard/zero_shot/{model}/scores.json"
        elif mode == "zero_shot":
            path = f"results/bongard/zero_shot/{model}/scores.json"
        elif mode == 100:
            path = f"results/bongard/with_solutions/{model}/scores.json"
        elif mode == 10:
            path = f"results/bongard/with_solutions_10/{model}/scores.json"
        elif mode == "attributes":
            path = f"results/bongard/with_attributes/{model}/scores.json"

        scores = json.load(open(path))

        n_solved_bps = [0] * len(list(scores.values())[0])

        counter = 0
        for bp_id, bp_scores in scores.items():

            if mode == "humans":
                # if model == "humans":
                #     df.loc[bp_id, model] = bp_scores
                # else:
                bp_scores = bp_scores
                num_solved = sum(bp_scores)
                df.loc[bp_id, model] = str(sum(bp_scores)) + "/" + str(len(bp_scores))

                values = df.loc[bp_id, model].split("/")
                int_values = [int(x) for x in values]
                ratio = int_values[0] / int_values[1]

                if ratio >= 0.33 and ratio < 0.66:
                    df.loc[bp_id, model] = (
                        "\cellcolor{YellowGreen!25}" + df.loc[bp_id, model]
                    )
                if ratio >= 0.66 and ratio < 1:
                    df.loc[bp_id, model] = "\cellcolor{Green!25}" + df.loc[bp_id, model]
                if ratio == 1:
                    df.loc[bp_id, model] = (
                        "\cellcolor{PineGreen!25}" + df.loc[bp_id, model]
                    )

            else:
                num_solved = sum(bp_scores)
                if num_solved == 0:
                    df.loc[bp_id, model] = "0/3"
                elif num_solved == 1:
                    df.loc[bp_id, model] = "\cellcolor{YellowGreen!25}1/3"
                elif num_solved == 2:
                    df.loc[bp_id, model] = "\cellcolor{Green!25}2/3"
                elif num_solved == 3:
                    df.loc[bp_id, model] = "\cellcolor{PineGreen!25}3/3"

            df_ratios.loc[bp_id, model] = num_solved / len(n_solved_bps)

            for run_id, score in enumerate(bp_scores):

                if score == 1:
                    n_solved_bps[run_id] += 1

            counter += 1

        print(f"Model: {model}")
        print(f"run 1: {n_solved_bps[0]}")
        print(f"run 2: {n_solved_bps[1]}")
        print(f"run 3: {n_solved_bps[2]}")

        # get mean
        mean_solved_bps = sum(n_solved_bps) / len(n_solved_bps)
        # get std
        n_solved_bps_std = np.std(n_solved_bps)

        print(f"Mean: {mean_solved_bps}")
        print(f"Std: {n_solved_bps_std}")

    # remove nan rows
    # df = df.dropna()

    # rename columns
    df.columns = [map_model_names(x) for x in df.columns]

    df["Categories"] = [get_categpry(int(x)) for x in df.index]

    # print df as latex table
    print(df.to_latex())


def get_correlation_between_models_and_humans():

    df = pd.read_csv(
        "results/for_plotting/single_bp_results_humans_and_models.csv", index_col=0
    )

    # drop first row
    df = df.drop(1)
    # get humans
    humans = df["humans"].values
    # get models
    models = df.columns[1:]

    for model in models:
        model_values = df[model].values

        # get correlation
        correlation = np.corrcoef(humans, model_values)[0, 1]

        print(f"Correlation between humans and {model}: {correlation}")

    # which bps were soilved by o1 but not by humans
    o1 = df["o1"]
    humans = df["humans"]

    bps = []
    for i in df["o1"].index.to_list():
        if o1[i] >= 50 and humans[i] < 50:
            print(f"BP {i} was solved by o1 but not by humans")
            bps.append(i)
    print("BPs solved by o1 but not by humans: ", len(bps))

    for i in df["o1"].index.to_list():
        if o1[i] < 50 and humans[i] >= 50:
            print(f"BP {i} was solved by humans but not by o1")
            bps.append(i)
    print("BPs solved by humans but not by o1: ", len(bps))


def number_of_bps_solved_at_least_once():

    # read df
    df = pd.read_csv(
        "results/for_plotting/single_bp_results_humans_and_models.csv", index_col=0
    )

    # drop first row
    df = df.drop(1)

    df = df.drop("humans.1", axis=1, errors="ignore")
    # get humans
    humans = df.columns[0]

    solved_bps_humans = [1 if x > 0 else 0 for x in df[humans].values]
    print(f"Number of BPs solved at least once by humans: {sum(solved_bps_humans)}")

    solved_by_all_humans = [1 if x == 100 else 0 for x in df[humans].values]
    print(f"Number of BPs solved by all humans: {sum(solved_by_all_humans)}")

    # get models
    models = df.columns[1:]

    solved_bps_across_models = []

    for model in models:
        model_values = df[model].values

        # get number of bps solved at least once
        solved_bps = [1 if x > 0 else 0 for x in model_values]
        solved_bps_across_models.append(solved_bps)

        print(f"Model: {model}")
        print(f"Number of BPs solved at least once: {sum(solved_bps)}")

    # get number of bps solved at least once by any model
    all_solved_bps_any_model = [
        1 if sum(x) > 0 else 0 for x in zip(*solved_bps_across_models)
    ]

    # get number of bps solved at least once by all models
    all_solved_bps_all_models = [
        1 if sum(x) == len(models) else 0 for x in zip(*solved_bps_across_models)
    ]

    print(
        f"Number of BPs solved at least once by any model: {sum(all_solved_bps_any_model)}"
    )
    print(
        f"Number of BPs solved at least once by all models: {sum(all_solved_bps_all_models)}"
    )


if __name__ == "__main__":
    models = [
        "humans_top_5",
        "o1",
        "gpt-4o",
        "claude-3-5-sonnet-20241022",
        "gemini-2.0-flash-exp",
        "gemini",
        "LlavaOnevision",
        "Qwen2VL",
        "InternVL2_5",
    ]

    # main(models, mode="humans")

    collect_agg_results(models, mode="humans")

    # get_correlation_between_models_and_humans()
    number_of_bps_solved_at_least_once()
