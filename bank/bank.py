from dataclasses import dataclass



@dataclass
class Bank:
    balance: float

    """This bank class manages all other bank related classes and methods to do with finance. The Trader Bank class is used to manage the bank account of each trader class individually."""

    def deposit(self, amount: float):
        self.balance += amount

    def withdraw(self, amount: float):
        self.balance -= amount

    def check_risk_limit(self, amount: float, risk_limit: float) -> bool:
        return amount <= risk_limit

    def apply_margin(self, amount: float, margin_rate: float) -> float:
        return amount * margin_rate

    def apply_fees(self, amount: float, fee_rate: float) -> float:
        return amount * fee_rate


#
@dataclass
class TraderBank:
    bank: Bank
    initial_balance: float
    balance: float
    margin_rate: float
    fee_rate: float

    """This Trader Bank class is used to manage the bank account of each trader class individually."""

    def __post_init__(self):
        self.bank = Bank(self.initial_balance)

    def deposit(self, amount: float):
        self.bank.deposit(amount)
        self.balance += amount

    def withdraw(self, amount: float):
        self.bank.withdraw(amount)
        self.balance -= amount

    def check_risk_limit(self, amount: float, risk_limit: float) -> bool:
        return self.bank.check_risk_limit(amount, risk_limit)

    def apply_margin(self, amount: float) -> float:
        return self.bank.apply_margin(amount, self.margin_rate)

    def apply_fees(self, amount: float) -> float:
        return self.bank.apply_fees(amount, self.fee_rate)
