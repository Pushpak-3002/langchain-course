from langsmith import traceable

@traceable
def test_trace():
    return "hello langsmith"

test_trace()