import pandas as pd


def eval(bp="neck"):

    if bp == "neck":
        left_side_answer = "horizontal"
        right_side_answer = "vertical"
    elif bp == "count":
        left_side_answer = "inside"
        right_side_answer = "outside"
    elif bp == "triangle_over_circle":
        left_side_answer = "triangle"
        right_side_answer = "circle"
    elif bp == "spirals":
        left_side_answer = "clockwise"
        right_side_answer = "counterclockwise"
    elif bp == "cavity":
        left_side_answer = "left"
        right_side_answer = "right"

    models = ["gpt-4o", "claude", "gemini", "llava_1.6"]
    df = pd.DataFrame(index=[str(x) for x in range(1, 12)], columns=models)

    for model in models:

        for i in range(12):

            correct_answers = [0, 0, 0]

            for idx, seed in enumerate([1, 2, 3]):

                path = f"results/{bp}/{model}/pe/{seed}/response_img_{i}.txt"

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
                    response.split(":")[-1].replace("'", "").replace('"', "").strip()
                )
                print(f"Answer: {answer}, Response: {response}")
                if answer == response:
                    correct_answers[idx] = 1

            # df.loc[str(i + 1), model] = str(sum(correct_answers)) + "/3"

            num_correct_answers = sum(correct_answers)
            if num_correct_answers == 0:
                df.loc[str(i + 1), model] = "0/3"
            elif num_correct_answers == 1:
                df.loc[str(i + 1), model] = "\cellcolor{YellowGreen!25}1/3"
            elif num_correct_answers == 2:
                df.loc[str(i + 1), model] = "\cellcolor{Green!25}2/3"
            elif num_correct_answers == 3:
                df.loc[str(i + 1), model] = "\cellcolor{PineGreen!25}3/3"

    print(df)

    print(df.transpose().to_latex())


if __name__ == "__main__":

    eval("neck")
    eval("count")
    eval("triangle_over_circle")
    eval("spirals")
