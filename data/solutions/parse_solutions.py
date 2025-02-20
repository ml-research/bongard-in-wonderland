from bs4 import BeautifulSoup
import json


def main():
    # parse file
    path = "data/solutions/Solutions of Bongard Problems - Harry Foundalis.html"

    with open(path, "r") as file:
        data = file.read()

    # parse html
    soup = BeautifulSoup(data, "html.parser")

    # get all tables
    tables = soup.find_all("table")

    # get all rows
    rows = tables[0].find_all("tr")

    bp_solutions = {}

    for id, row in enumerate(rows):
        # get all cells
        cells = row.find_all("td")

        bp_solutions[id + 1] = [
            cells[1].get_text().replace("\n", "").replace("\t\t\t\t", " "),
            cells[2].get_text().replace("\n", "").replace("\t\t\t\t", " "),
        ]

    # save solutions
    with open("data/solutions/bp_solutions.json", "w") as file:
        json.dump(bp_solutions, file)


if __name__ == "__main__":
    main()
