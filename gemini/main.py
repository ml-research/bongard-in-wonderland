import google.generativeai as genai


class Gemini:
    def __init__(self):
        self.load_api_key()
        self.model = genai.GenerativeModel("gemini-1.5-pro")

    def load_api_key(self):
        # read the API key from txt file
        path = "gemini/key.txt"
        with open(path, "r") as file:
            GOOGLE_API_KEY = file.read().strip()

        genai.configure(api_key=GOOGLE_API_KEY)

    def prompt(self, prompt_text):
        response = self.model.generate_content(prompt_text)
        return response

    def prompt_with_images(
        self, prompt_text, image_paths, system_prompt=None, seed=None
    ):
        image_path = image_paths[0]
        myfile = genai.upload_file(image_path)
        print(f"{myfile=}")

        if system_prompt is None:
            raise ValueError("system_prompt must be provided")

        result = self.model.generate_content(
            [myfile, "\n\n", system_prompt, "\n\n", prompt_text]
        )
        print(f"{result.text=}")
        return result.text


def main():
    gemini = Gemini()

    prompt_path = "prompts/spirals/spiral_base.txt"
    prompt_text = open(prompt_path, "r").read()
    image_path = ["data/bongard-problems-high-res/p0016/0.png"]

    response = gemini.prompt_with_images(prompt_text, image_path)
    print(f"{response=}")


if __name__ == "__main__":
    main()
