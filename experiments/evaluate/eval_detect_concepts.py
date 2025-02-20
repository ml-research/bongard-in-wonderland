import pandas as pd
import json


def eval():

    concept_options_path = "prompts/concepts/contrary_concepts.json"

    with open(concept_options_path, "r") as file:
        concept_options = json.load(file)

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

    df_of_concepts = pd.DataFrame(index=[str(x) for x in range(1, 100)], columns=models)

    df_of_concepts_two_third = pd.DataFrame(
        index=[str(x) for x in range(1, 100)], columns=models
    )

    correct_concepts = {}  # [0] * 101
    number_wrong_concepts = {}

    for model in models:
        correct_concepts[model] = [0] * 101
        number_wrong_concepts[model] = [0] * 101

    all_dfs = []

    for id, value in concept_options.items():

        left_side_answer = value[0]
        right_side_answer = value[1]

        df = pd.DataFrame(index=[str(x) for x in range(1, 12)], columns=models)

        for model in models:

            all_correct = True
            all_two_third_correct = True

            correct_answers_over_run = 0

            for i in range(12):

                if model == "o1":
                    correct_answers = [0]
                else:
                    correct_answers = [0, 0, 0]

                for idx, run in enumerate([1, 2, 3]):

                    path = (
                        f"results/concepts/{model}/BP_{id}/{run}/response_img_{i}.txt"
                    )

                    try:
                        with open(path, "r") as file:
                            response = file.read()

                        # parse string between {}
                        response = response.split("{")[1].split("}")[0]

                    except:
                        print(f"Could not read response from {path}")
                        response = "error"

                    if i < 6:
                        answer = left_side_answer
                        wrong_answer = right_side_answer
                    else:
                        answer = right_side_answer
                        wrong_answer = left_side_answer

                    response = (
                        response.split(":")[-1]
                        .replace("'", "")
                        .replace('"', "")
                        .strip()
                    )
                    print(f"Answer: {answer}, Response: {response}")
                    if answer == response:
                        correct_answers[idx] = 1
                        correct_answers_over_run += 1

                # df.loc[str(i + 1), model] = str(sum(correct_answers)) + "/3"

                if model == "o1":
                    correct_answers = correct_answers * 3
                assert len(correct_answers) == 3
                num_correct_answers = sum(correct_answers)

                # if 2/3 is once not correct, set all_two_third_correct to False
                if num_correct_answers < 2:
                    all_two_third_correct = False

                if num_correct_answers < 3:
                    all_correct = False
                    number_wrong_concepts[model][int(id)] += 1

                if num_correct_answers == 0:
                    df.loc[str(i + 1), model] = "0/3"
                elif num_correct_answers == 1:
                    df.loc[str(i + 1), model] = "\cellcolor{YellowGreen!25}1/3"
                elif num_correct_answers == 2:
                    df.loc[str(i + 1), model] = "\cellcolor{Green!25}2/3"
                elif num_correct_answers == 3:
                    df.loc[str(i + 1), model] = "\cellcolor{PineGreen!25}3/3"

            num_correct_answers = 12 - number_wrong_concepts[model][int(id)]
            # df_of_concepts.loc[str(id), model] = num_correct_answers / 12
            if model == "o1":
                df_of_concepts.loc[str(id), model] = correct_answers_over_run / 12
            else:
                df_of_concepts.loc[str(id), model] = correct_answers_over_run / 36

            df_of_concepts_two_third.loc[str(id), model] = (
                1 if all_two_third_correct else 0
            )

            if all_correct:
                correct_concepts[model][int(id)] += 1

        all_dfs.append(df)

        # print(df)

        # print(df.transpose().to_latex())

    print(correct_concepts)

    for bp_id in [16, 55, 29, 37]:
        print(f"BP {bp_id}:")
        print(all_dfs[bp_id - 1].transpose().to_latex(float_format="%.2f"))

    print("-----------------------------\n")

    for model in models:
        print(f"\n---------- Model: {model} --------------\n")
        # number of correct concepts
        print("Number of correct concepts: ", sum(correct_concepts[model]))

        # get ids of correct concepts
        correct_concept_ids = [
            idx for idx, value in enumerate(correct_concepts[model]) if value > 0
        ]
        print(f"BPs with correct concepts for model {model}: ", correct_concept_ids)

        # number of wrong concepts
        print(
            f"Number of wrong concepts for model {model}: ",
            number_wrong_concepts[model],
        )

        # BP ids with 6 or more wrong concepts
        wrong_concept_ids = [
            idx for idx, value in enumerate(number_wrong_concepts[model]) if value >= 6
        ]
        print(
            f"BPs with 6 or more wrong concepts for model {model}: ", wrong_concept_ids
        )

    print(df_of_concepts)

    # print latex table with rounded values
    print(df_of_concepts.to_latex(float_format="%.2f"))

    # save df as csv
    df_of_concepts.to_csv(f"results/concepts/summary.csv")

    df_of_concepts_two_third.to_csv(f"results/concepts/summary_two_third.csv")


def compare_correct_concepts_to_correct_bps():

    df_of_concepts = pd.read_csv("results/concepts/summary.csv", index_col=0)
    df_of_solved_bps = pd.read_csv(
        "results/for_plotting/single_bp_results_humans_and_models.csv", index_col=0
    )

    models = df_of_concepts.columns

    for model in models:

        correct_concepts = df_of_concepts[model].values
        correct_bps = df_of_solved_bps[model].values

        print(f"Model: {model}")

        correct_concepts = set(
            [idx + 1 for idx, value in enumerate(correct_concepts) if value >= 1]
        )
        correct_bps = set(
            [idx + 1 for idx, value in enumerate(correct_bps) if value > 0.3]
        )

        print(f"Correct concepts: {len(correct_concepts)}")
        print(f"Correct BPs: {len(correct_bps)}")

        print(f"Correct concepts: {correct_concepts}")
        print(f"Correct BPs: {correct_bps}")

        print(f"Correct concepts and BPs: {correct_concepts.intersection(correct_bps)}")
        print(
            f"Number of correct concepts and BPs: {len(correct_concepts.intersection(correct_bps))}"
        )
        correct_concepts_but_not_bps = list(correct_concepts - correct_bps)
        correct_concepts_but_not_bps.sort()

        print(f"Correct concepts but not BPs: {correct_concepts_but_not_bps}")
        print(
            f"Number of correct concepts but not BPs: {len(correct_concepts - correct_bps)}"
        )
        # print(f"Correct BPs but not concepts: {correct_bps - correct_concepts}")
        # print(
        #     f"Number of correct BPs but not concepts: {len(correct_bps - correct_concepts)}"
        # )

        print("\n--------------------------------------------------\n")


def get_accuracy_distributions_of_solved_bps():

    df_of_solved_bps = pd.read_csv(
        "results/for_plotting/single_bp_results_humans_and_models.csv", index_col=0
    )
    df_of_concepts = pd.read_csv("results/concepts/summary.csv", index_col=0)

    models = df_of_solved_bps.columns

    for model in models:

        if "humans" in model:
            continue

        accuracies = []

        correct_concepts = df_of_concepts[model].values
        correct_bps = df_of_solved_bps[model].values

        correct_bps = df_of_solved_bps[model].values
        correct_bps = [idx + 1 for idx, value in enumerate(correct_bps) if value > 30]

        correct_bps.sort()

        for bp_id in correct_bps:

            correct_concepts_for_bp = df_of_concepts.loc[bp_id, model]
            accuracies.append(correct_concepts_for_bp)

        print(f"Model: {model}")
        print(accuracies)
        print("\n--------------------------------------------------\n")

        # plot accuracies as histogram
        import matplotlib.pyplot as plt

        plt.hist(accuracies, bins=10)
        plt.xlabel("Accuracy")
        plt.ylabel("Frequency")
        plt.title(f"Accuracy distribution for model {model}")
        plt.show()


def compare_q1_q3_q4():

    df_of_solved_bps = pd.read_csv(
        "results/for_plotting/single_bp_results_humans_and_models.csv", index_col=0
    )
    df_of_concepts = pd.read_csv("results/concepts/summary.csv", index_col=0)
    df_of_concepts_two_third = pd.read_csv(
        "results/concepts/summary_two_third.csv", index_col=0
    )
    df_of_hypotheses = pd.read_csv("results/hypotheses/all_results.csv", index_col=0)

    models = df_of_solved_bps.columns

    solved_across_models = set(df_of_solved_bps.index)
    unsolved_across_models = set(df_of_solved_bps.index)
    unsolved_across_models_count = []

    for model in models:

        if "humans" in model:
            continue

        # get BPs that are not solved by the model
        unsolved_bps_q1 = [
            idx + 1
            for idx, value in enumerate(df_of_solved_bps[model].values)
            if value <= 30
        ]

        unsolved_bps_q3 = [
            idx + 1
            for idx, value in enumerate(df_of_concepts[model].values)
            if value <= 1
        ]

        solved_bps_q3_two_third = [
            idx + 1
            for idx, value in enumerate(df_of_concepts_two_third[model].values)
            if value == 1
        ]

        unsolved_q3_two_third = [
            idx + 1
            for idx, value in enumerate(df_of_concepts_two_third[model].values)
            if value == 0
        ]

        unsolved_bps_q4 = [
            idx + 1
            for idx, value in enumerate(df_of_hypotheses[model].values)
            if value == 0
        ]

        solved_across_models = solved_across_models - set(unsolved_bps_q1)
        solved_across_models = solved_across_models - set(unsolved_bps_q3)
        solved_across_models = solved_across_models - set(unsolved_bps_q4)

        print(f"Model: {model}")

        print(f"Solved BPs Q3 2/3: {len(solved_bps_q3_two_third)}")

        # get intersection of unsolved bps
        unsolved_bps = (
            set(unsolved_bps_q1)
            .intersection(unsolved_q3_two_third)
            .intersection(unsolved_bps_q4)
        )
        unsolved_bps = list(unsolved_bps)
        unsolved_bps.sort()
        print(f"Unsolved BPs: {unsolved_bps}")

        unsolved_across_models = unsolved_across_models.intersection(unsolved_bps)
        unsolved_across_models_count.append(unsolved_bps)

    unsolved_across_models = list(unsolved_across_models)
    unsolved_across_models.sort()
    print(f"Unsolved BPs across all models: {unsolved_across_models}")

    # get bps unsolved at least t times
    t = 6
    unsolved_across_models_count_three = []
    for i in range(1, 101):
        count = 0
        for l in unsolved_across_models_count:
            if i in l:
                count += 1
        if count >= t:
            unsolved_across_models_count_three.append(i)

    print(
        f"Unsolved BPs across all models at least {t} times: {unsolved_across_models_count_three}"
    )

    print(f"Solved BPs across all models: {solved_across_models}")


def table_8():

    df_of_solved_bps = pd.read_csv(
        "results/for_plotting/single_bp_results_humans_and_models.csv", index_col=0
    )
    df_of_concepts = pd.read_csv("results/concepts/summary.csv", index_col=0)
    df_of_concepts_two_third = pd.read_csv(
        "results/concepts/summary_two_third.csv", index_col=0
    )
    df_of_hypotheses = pd.read_csv("results/hypotheses/all_results.csv", index_col=0)

    models = df_of_solved_bps.columns

    solved_across_models = set(df_of_solved_bps.index)
    unsolved_across_models = set(df_of_solved_bps.index)
    unsolved_across_models_count = []

    for model in models:

        if "humans" in model:
            continue

        # get BPs that are not solved by the model
        solved_bps_q1 = [
            idx + 1
            for idx, value in enumerate(df_of_solved_bps[model].values)
            if value >= (2 / 3)
        ]

        solved_bps_q3_two_third = [
            idx + 1
            for idx, value in enumerate(df_of_concepts_two_third[model].values)
            if value == 1
        ]

        # q1 - q3
        solved_bps_q1_not_q3 = list(set(solved_bps_q1) - set(solved_bps_q3_two_third))
        num_solved_bps_q1_not_q3 = len(solved_bps_q1_not_q3)

        # solved bps in q1 and q3
        solved_bps_q1_and_q3 = list(
            set(solved_bps_q1).intersection(set(solved_bps_q3_two_third))
        )

        num_solved_q1 = len(solved_bps_q1)
        ratio_solved_q1_and_q3 = len(solved_bps_q1_and_q3) / num_solved_q1

        print(f"Model: {model}")
        print(f"Number of solved BPs in Q1: {num_solved_q1}")
        print(f"Number of solved BPs in Q3: {len(solved_bps_q3_two_third)}")
        print(f"Number of solved BPs in Q1 and Q3: {len(solved_bps_q1_and_q3)}")
        # print(f"Number of solved BPs in Q1 but not in Q3: {num_solved_bps_q1_not_q3}")
        print(
            f"Ratio of solved BPs in Q1 but not in Q3: {ratio_solved_q1_and_q3 * 100:.2f}\%"
        )


def table_10():

    df_of_solved_bps = pd.read_csv(
        "results/for_plotting/single_bp_results_humans_and_models.csv", index_col=0
    )
    df_of_concepts_two_third = pd.read_csv(
        "results/concepts/summary_two_third.csv", index_col=0
    )
    df_of_hypotheses = pd.read_csv("results/hypotheses/all_results.csv", index_col=0)

    models = df_of_solved_bps.columns

    for model in models:

        if "humans" in model:
            continue

        # get BPs that are not solved by the model
        solved_bps_q1 = [
            idx + 1
            for idx, value in enumerate(df_of_solved_bps[model].values)
            if value >= ((2 / 3) * 100)
        ]

        solved_bps_q3 = [
            idx + 1
            for idx, value in enumerate(df_of_concepts_two_third[model].values)
            if value >= 1
        ]

        solved_bps_q4 = [
            idx + 1
            for idx, value in enumerate(df_of_hypotheses[model].values)
            if value == 1
        ]

        solved_q1_and_q4 = set(solved_bps_q1).intersection(set(solved_bps_q4))
        ratio_solved_q1_and_q4 = len(solved_q1_and_q4) / len(solved_bps_q1)

        solved_bps_q1 = set(solved_bps_q1)
        solved_bps_q3 = set(solved_bps_q3)
        solved_bps_q4 = set(solved_bps_q4)

        type_i = solved_bps_q1.intersection(solved_bps_q3).difference(solved_bps_q4)
        type_ii = solved_bps_q3.intersection(solved_bps_q4).difference(solved_bps_q1)
        type_iii = solved_bps_q1.intersection(solved_bps_q4).difference(solved_bps_q3)
        type_iv = solved_bps_q1.intersection(solved_bps_q3).intersection(solved_bps_q4)

        print(f"Model: {model}")
        print(f"Number of solved BPs in Q4: {len(solved_bps_q4)}")
        print(f"Number of solved BPs in Q1 and Q4: {len(solved_q1_and_q4)}")
        print(f"Number of solved BPs in Q1: {len(solved_bps_q1)}")
        print(f"Ratio of solved BPs in Q1 and Q4: {ratio_solved_q1_and_q4 * 100:.2f}\%")
        # print(f"Type I: {len(type_i)}")
        # print(f"Type II: {len(type_ii)}")
        # print(f"Type III: {len(type_iii)}")
        # print(f"Type IV: {len(type_iv)}")
        print(
            f"Types: \t\t {len(type_i)} & {len(type_ii)} & {len(type_iii)} & {len(type_iv)}"
        )

        print(f"Type IV / Q1: {len(type_iv) / len(solved_bps_q1) * 100:.2f}\%")


if __name__ == "__main__":

    # eval()
    # compare_correct_concepts_to_correct_bps()
    # get_accuracy_distributions_of_solved_bps()

    # which bps are especially hard?
    # compare_q1_q3_q4()

    table_10()
