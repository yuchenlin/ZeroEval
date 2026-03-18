from src.evaluation.hendrycks_math_utils import is_equiv, check_string_is_number
import pytest

test_data = [
 ("200.0", "200", True),
 ("200", "200", True),
 ("5.1", "5.1", True),
 ("5.10", "5.1", True),
 ("5.11", "5.1", False),
 ("-2x", "-2x", True),
 ("ellipse", r"\text{ellipse}", True),
 ("(B)", r"\boxed{(B)}", True),
 ("(B)", r"\textbf{(B)}", True),
 ("4343.0", "4343_0", False),
 ("864.0", r"864\mbox{inches}^2", True), 
 ("30", r"30^\circ", True), 
]

@pytest.mark.parametrize("a,b,expected", test_data)
def test_is_equiv(a: str, b:str, expected: bool):
    is_correct = is_equiv(a,b, verbose=True)
    assert is_correct == expected 


test_data = [
 ("200_1", False),
 ("5.1",True),
 ("5",True),
 ("-2x",  False),
]

@pytest.mark.parametrize("a,expected", test_data)
def test_is_number(a, expected: bool):
    is_correct = check_string_is_number(a)
    assert is_correct==expected    