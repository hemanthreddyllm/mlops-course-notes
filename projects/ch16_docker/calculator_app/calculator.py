"""Pure calculator logic, kept separate from the web layer so it can be tested on its own."""

OPERATIONS = {
    "add": ("+", lambda a, b: a + b),
    "subtract": ("−", lambda a, b: a - b),
    "multiply": ("×", lambda a, b: a * b),
    "divide": ("÷", lambda a, b: a / b),
}


def calculate(a: float, b: float, operation: str) -> float:
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation: {operation}")
    if operation == "divide" and b == 0:
        raise ZeroDivisionError("cannot divide by zero")
    return OPERATIONS[operation][1](a, b)
