def calculate_position_size(entry_price: float, stop_price: float, risk_amount: float) -> dict:
    """
    Calculate the position size and related metrics based on entry price, stop loss, and risk amount.
    
    Parameters:
    entry_price (float): The price at which you plan to enter the trade
    stop_price (float): Your stop loss price
    risk_amount (float): The amount of money you're willing to risk on this trade
    
    Returns:
    dict: Dictionary containing position details including:
        - shares: Number of shares to purchase
        - total_value: Total position value
        - risk_percentage: Percentage drop to stop loss
        - price_per_share: Entry price per share
        - potential_loss: Total loss if stop is triggered
    """
    if entry_price <= 0 or stop_price <= 0 or risk_amount <= 0:
        raise ValueError("All input values must be positive numbers")
    
    if stop_price >= entry_price:
        raise ValueError("Stop price must be below entry price for long positions")
    
    # Calculate the price difference and risk percentage
    price_difference = entry_price - stop_price
    risk_percentage = (price_difference / entry_price) * 100
    
    # Calculate position size based on risk
    shares = int(risk_amount / price_difference)
    
    # Calculate total position value and potential loss
    total_position_value = shares * entry_price
    total_loss_at_stop = shares * (entry_price - stop_price)
    
    return {
        "shares": shares,
        "total_value": round(total_position_value, 2),
        "risk_percentage": round(risk_percentage, 2),
        "price_per_share": entry_price,
        "potential_loss": round(total_loss_at_stop, 2),
        "entry_price": entry_price,
        "stop_price": stop_price,
        "risk_amount": risk_amount
    }

def print_position_summary(symbol, position_details: dict,  notes:str='') -> None:
    """
    Print a formatted summary of the position details.
    """
    print("  ")
    print(f'========= - {symbol} - Summary ======================')
    print(f"Number of shares to purchase : {position_details['shares']:,}")
    print(f"Price per share              : ${position_details['price_per_share']:.2f}")
    print(f"Total position value         : ${position_details['total_value']:,.2f}")
    print(f"Risk percentage to stop      : {position_details['risk_percentage']}%")
    print(f"Potential loss at stop       : ${position_details['potential_loss']:,.2f}")
    print(" -------------------------------------------------")
    print(f"Entry Price                 : {position_details['entry_price']}")
    print(f"Stop Price                  : {position_details['stop_price']}")
    print(f"Risk Amount                 : {position_details['risk_amount']}")
    print(" -------------------------------------------------")
    print(f"Notes                        : {notes}")
    print("=====================")