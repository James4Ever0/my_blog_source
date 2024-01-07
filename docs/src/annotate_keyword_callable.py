
from typing import Callable

def myfunction(a: int, b: int = 1) -> int:
    return 1

# Annotating myfunction as Callable
myfunction: Callable[[int, int], int]