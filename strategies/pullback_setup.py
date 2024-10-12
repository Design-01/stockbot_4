from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class PullbackSetup:
    times: List[Tuple[str, str]]
    barSizes: List[str]
    ls: List[str]
    minAvPass: int
    maxFail: int
    minAvScore: float
    signals: List[Tuple[str, Tuple[float, float], float]]

    def calculate(self, data):
        # Placeholder for PullbackSetup calculation
        pass
