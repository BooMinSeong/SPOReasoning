from typing import List,  Dict, Any
from datasets import Dataset
from math_eval import extract_answer, math_equal


# For Dataset map
def extract_completion_answers(
    x: Dict[str, Any]
) -> Dict[str, List[str]]:
    parsed_answers = []
    completions_input = x.get("completions")

    if isinstance(completions_input, list):
        for item in completions_input:
            if isinstance(item, str):
                parsed_answers.append(extract_answer(item, "math"))
            else:
                # 리스트 내의 항목이 문자열이 아닌 경우, 빈 문자열로 처리
                parsed_answers.append(extract_answer("", "math"))
    elif isinstance(completions_input, str):
        parsed_answers.append(extract_answer(completions_input, "math"))
    # completions_input이 None이거나 다른 타입이면 parsed_answers는 비어 있게 됩니다.

    return {"completion_parsed_answers": parsed_answers} # Key is now plural "completion_parsed_answers"


def match_answers(
    x: Dict[str, Any]
) -> Dict[str, List[bool]]:
    is_correct = []
    parsed_answers_list = x.get("completion_parsed_answers")
    correct_response = x.get("answer")

    if isinstance(parsed_answers_list, list) and \
       correct_response:  # 정답 리스트가 비어있지 않은지 확인

        correct_target = correct_response  # 첫 번째 정답을 기준으로 비교
        for parsed_answer in parsed_answers_list:
            if math_equal(parsed_answer, correct_target):
                is_correct.append(True)
            else:
                is_correct.append(False)
                
    return {"completions_is_correct": is_correct} # Key name can remain "completions_is_correct"


def score(dataset:Dataset) -> Dataset:
    dataset = dataset.map(
        extract_completion_answers,
        desc = "Extract all parsed answers from completions" # Updated description
    )
    dataset = dataset.map(
        match_answers,
        desc = "Evaluate if any parsed answer is correct" # Updated description
    )
    
    return dataset