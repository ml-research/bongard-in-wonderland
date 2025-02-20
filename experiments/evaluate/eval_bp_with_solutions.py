import json
import argparse


def parse_response(response_content):

    if "{" not in response_content:
        return -1

    try:
        # parse dict from response_content
        response_content = "{" + response_content.split("{")[-1].split("}")[0] + "}"
    except:
        # raise ValueError(
        #     f"Could not parse dict from response_content: {response_content}"
        # )
        return -1

    # change ' to " for json parsing
    response_content = response_content.replace("\n", "")
    response_content = response_content.replace('"', "'")
    response_content = response_content.replace(" '", ' "')
    response_content = response_content.replace("' ", '" ')
    response_content = response_content.replace("',", '",')
    response_content = response_content.replace("':", '":')
    response_content = response_content.replace("'}", '"}')
    response_content = response_content.replace("{'", '{"')
    response_content = response_content.replace(",", "")

    response_content = response_content.replace("None", "-1")

    # if there is a comment in the response, remove it
    if "#" in response_content:
        response_content = response_content.split("#")[0] + "}"

    try:
        response_content = json.loads(response_content)
    except:
        # raise ValueError(
        #     f"Could not parse json from response_content: {response_content}"
        # )
        print(f"Could not parse json from response_content: {response_content}")
        return -1

    try:
        response_content = response_content["answer"]
    except:
        return -1

    return response_content


def evaluate(model, mode=100):

    scores = {}
    if mode == 100:
        path = f"results/bongard/with_solutions/{model}"
    elif mode == 10:
        path = f"results/bongard/with_solutions_10/{model}"

    for i in range(1, 101):

        scores[i] = []
        for run in [1, 2, 3]:

            response_path = f"{path}/BP_{i}/response_run_{run}.txt"
            with open(response_path, "r") as file:
                response = file.read()

            # parse answer from response
            response_content = parse_response(response)

            if type(response_content) != int:
                try:
                    response_content = int(response_content)
                except:
                    response_content = -1

            # get score
            if response_content == i:
                score = 1
            elif i in [6, 10, 96, 98]:
                if response_content in [6, 10, 96, 98]:
                    score = 1
            elif i in [85, 86, 88, 89]:
                if response_content in [85, 86, 88, 89]:
                    score = 1
            else:
                score = 0

            scores[i].append(score)

    # save scores
    scores_path = f"{path}/scores.json"
    with open(scores_path, "w") as file:
        json.dump(scores, file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--mode", type=int, default=10)

    args = parser.parse_args()

    evaluate(args.model, mode=args.mode)
