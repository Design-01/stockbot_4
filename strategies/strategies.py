from dataclasses import dataclass

@dataclass
class EntryStrategy:
    name: str
    parameters: dict

    def apply(self, market_data):
        raise NotImplementedError("This method should be overridden by subclasses")

@dataclass
class ExitStrategy:
    name: str
    parameters: dict

    def apply(self, market_data):
        raise NotImplementedError("This method should be overridden by subclasses")