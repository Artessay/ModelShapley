import re


def extract_answer(text, method="strict"):
    assert method in ["strict", "flexible"]

    pattern = r"answer is \(?([A-Z])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    elif method == "strict":
        return 'Z'
    else:
        return _extract_again(text)


def _extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-Z])', text)
    if match:
        return match.group(1)
    else:
        return _extract_next(text)

def _extract_next(text):
    match = re.search(r'\(([A-Z])\)', text)
    if match:
        return match.group(0)
    else:
        return _extract_final(text)


def _extract_final(text):
    pattern = r"\b[A-Z]\b(?!.*\b[A-Z]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return 'Z'


def compute_score(solution_str, ground_truth, method="flexible"):
    """The scoring function for MMLU.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_answer(solution_str, method=method)
    if answer == ground_truth:
        return 1.0
    else:
        return 0.0
