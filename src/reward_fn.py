from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
import re

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = completions
    rewards = []
    i = 0
    for content, sol in zip(contents, solution):
        gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
            # if reward == 0:
            #     print(f"\033[91m{kwargs['problem'][i]}\n{content}\n ->  {answer_parsed} | {gold_parsed}\033[0m\n================================================\n")
            # else:
            #     print(f"\033[92m{kwargs['problem'][i]}\n{content}\033[0m\n================================================\n")
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
        i += 1
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^\s*<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$"
    completion_contents = [completion for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

if __name__ == "__main__":

    completions_0 = """
<think>

... some text ...
</think>

<answer>
maybe

Thus, the final answer is \(\boxed{0}\).

</answer>
"""
    completions_1 = """<think>....haha</think><answer>maybe ... \(\boxed{0}\)</answer>"""
    completions_2 = """<think>....haha maybe</think><answer>maybe ..."""
    completions_3 = """<think>....haha maybe. sin(60.."""

    print(format_reward([completions_0, completions_1, completions_2, completions_3]))