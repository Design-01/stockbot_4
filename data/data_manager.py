import pandas as pd
from dataclasses import dataclass, field
from typing import List
from strategies.ta import Indicator

@dataclass
class DataManager:
    data: pd.DataFrame
    indicators: List[Indicator] = field(default_factory=list)

    def add_ta(self, indicator: Indicator):
        self.indicators.append(indicator)
        result = indicator.run(self.data)
        if isinstance(result, pd.Series):
            self.data = pd.concat([self.data, result], axis=1)
            return result.name
        else:
            self.data = pd.concat([self.data, result], axis=1)
            return result.columns


    def update_data(self, new_row: pd.Series):
        self.data = pd.concat([self.data, new_row.to_frame().T])
        self._update_indicators()

    def _update_indicators(self):
        for indicator in self.indicators:
            result = indicator.run(self.data)
            if isinstance(result, pd.Series):
                self.data[result.name] = result
            else:
                for col in result.columns:
                    self.data[col] = result[col]

    def get_data(self, names: List[str] | str):
        if isinstance(names, str):
            return self.data[names]
        return self.data[names]
