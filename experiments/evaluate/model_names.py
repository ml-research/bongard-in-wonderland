def map_model_names(name):

    orig_name = name
    name = name.lower()

    if "o1" in name:
        return "o1"

    if "gpt-4o" in name:
        return "GPT-4o"

    if "claude" in name:
        return "Claude 3.5"

    if "gemini" in name and "flash" in name:
        return "Gemini 2.0"

    if "gemini" in name:
        return "Gemini 1.5"

    if "llava" in name:
        return "LlaVA-OV"

    if "qwen" in name:
        return "Qwen2VL"

    if "intern" in name:
        return "InternVL 2.5"

    if "human" in name:
        if "5" in name:
            return "Human (Top 5)"
        return "Human"

    return orig_name
