# Bongard in Wonderland
This is the official repository of the article ["Bongard in Wonderland: Visual Puzzles that Still Make AI Go Mad?"](https://openreview.net/pdf?id=4Yv9tFHDwX). 

## Getting started
To run the code you can either set up a conda environment and install `requirements.txt` (without LLaVA) or build the docker container to launch LLaVA on your machine. You can find more details on that in `llava_steps.md`. 

## Usage
The experimental scripts can be found in `experiments/`. You can execute them from the command line, e.g.,
```bash
python experiments/zero_shot_bp.py --model "gpt-4o"
```
Make sure to include your API access keys in the respective folders of the model, e.g., `gpt-4o/open-ai-key`.

The results of the evaluations will be stored in `results/`. The evaluation scripts, including the llm-judge can be found in `experiments/evaluate`. You can run those from the command line as well, e.g.,
```bash
python experiments/zero_shot_bp.py --model "gpt-4o" --mode "zero_shot"
```

## Data
We use the dataset provided by Depeweg et. al [1] which contains the 100 original Bongard Problems in high resolution ([Link here](https://osf.io/95dks/)). 
For the perception-focussed evaluation we considered the single diagrams of BPs 16, 19, 29 and 36. These are stored in `data/bongard-problems-high-res/`.

[1] Depeweg, S., Rothkopf, C.A., Jäkel, F. (2024). [Solving Bongard Problems with a Visual Language and Pragmatic Constraints](https://onlinelibrary.wiley.com/doi/10.1111/cogs.13432). Cognitive Science, 48(5), e13432.