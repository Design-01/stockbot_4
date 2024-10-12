from dataclasses import dataclass
from typing import List

@dataclass
class Stop:
    implement_when_rr: float
    stop_type: str
    offset: float

@dataclass
class Target:
    risk_reward: float

@dataclass
class ExitStrategy:
    stops: List[Stop]
    target: Target

    def calculate(self, data):
        # Placeholder for ExitStrategy calculation
        pass
