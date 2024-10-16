import glob

import sglang as sgl
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json


class LlaVaPrompter:

    def __init__(self, endpoint="http://127.0.0.1:10000"):

        # set the default backend
        sgl.set_default_backend(sgl.RuntimeEndpoint(endpoint))

        # set the regex for the answers
        self.example_regex_for_answers = (
            r"""\{\n"""
            + r"""    "set A rule": "[\w\d\s]{1,500}",\n"""
            + r"""    "set B rule": "[\w\d\s]{1,500}",\n"""
            + r"""\}"""
        )

    def prompt(self, prompt_text, system_prompt, seed=None):
        pass

    @sgl.function
    def bg_gen_no_regex(s, system_prompt, text_prompt, image_path):

        # s += sgl.system(system_prompt)
        if system_prompt is not None:
            s += sgl.user(system_prompt)
        s += sgl.user(sgl.image(image_path) + text_prompt)
        hyperparameters = {"temperature": 0.2, "top_p": 0.95, "top_k": 50}
        s += sgl.assistant(
            sgl.gen("inductive_logic", max_tokens=1000, **hyperparameters)
        )

    def prompt_with_images(
        self, prompt_text: str, paths: [str], system_prompt=None, seed=None
    ):

        if len(paths) == 1:
            path = paths[0]
        else:
            raise NotImplementedError("Only one image is supported at the moment.")

        states = self.bg_gen_no_regex.run_batch(
            [
                {
                    "system_prompt": system_prompt,
                    "text_prompt": prompt_text,
                    "image_path": path,
                }
            ],
            temperature=0.2,
            progress_bar=True,
        )
        print(states[0]["inductive_logic"])
        return states[0]["inductive_logic"]
