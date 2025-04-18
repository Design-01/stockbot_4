import functools
import warnings
from typing import Callable, List, TypeVar, cast

# Define a generic return type
R = TypeVar('R')

def validate_range(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator that validates if all values in the input list are between 0 and 1.
    
    This decorator checks the first argument of the decorated function (expected to be 
    a list of floats) and issues a warning if any values fall outside the range [0, 1].
    
    Args:
        func: The function to be decorated
        
    Returns:
        The decorated function with input validation
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if first argument is a list/iterable
        if args and hasattr(args[0], '__iter__'):
            values = args[0]
            out_of_range = [v for v in values if not (0 <= v <= 1)]
            
            if out_of_range:
                warnings.warn(
                    f"Function {func.__name__} received {len(out_of_range)} values outside "
                    f"the range [0, 1]. This may lead to skewed results. "
                    f"Out-of-range values: {out_of_range[:5]}{'...' if len(out_of_range) > 5 else ''}"
                )
        
        # Call the original function
        return func(*args, **kwargs)
    
    return cast(Callable[..., R], wrapper)


"""
Scoring Module

This module provides a collection of scoring functions that combine multiple values using 
different weighting strategies. Each function implements a specific scoring methodology
suitable for different use cases, from simple weighted averages to more complex combinations
of maximum values and weighted means.

Input values are expected to be in the range [0, 1] for consistent results.
"""


@validate_range
def max_plus_mean(vals: list[float], maxWeight: float) -> float:
    """Calculate score based on the lists of floats. Gets max value, gets mean of remaining values.
    Adds them together using the weight of the max value to portion the sum of the two.
    
    Args:
        vals: List of float values to score
        maxWeight: Weight for the maximum value (between 0 and 1)
        
    Returns:
        Final weighted score
        
    Example:
        If maxWeight is 0.8, max value accounts for 80% of the total score
    """
    if not vals:
        return 0.0
    
    max_val = max(vals)
    remaining_vals = [v for v in vals if v != max_val]
    
    # If only one value in the list or all values are the same
    if not remaining_vals:
        return max_val
    
    mean_remaining = sum(remaining_vals) / len(remaining_vals)
    
    # Combine with weights
    return max_val * maxWeight + mean_remaining * (1 - maxWeight)


@validate_range
def max_plus_max(vals: list[float], maxWeight: float) -> float:
    """Calculate score based on the lists of floats. Gets max value, gets max of remaining values.
    Adds them together using the weight of the max value to portion the sum of the two.
    
    Args:
        vals: List of float values to score
        maxWeight: Weight for the maximum value (between 0 and 1)
        
    Returns:
        Final weighted score
        
    Example:
        If maxWeight is 0.8, max value accounts for 80% of the total score
    """
    if not vals:
        return 0.0
    
    max_val = max(vals)
    remaining_vals = [v for v in vals if v != max_val]
    
    # If only one value in the list
    if not remaining_vals:
        return max_val
    
    second_max = max(remaining_vals)
    
    # Combine with weights
    return max_val * maxWeight + second_max * (1 - maxWeight)


@validate_range
def weighted_top_n_average(vals: list[float], weights: list[float]) -> float:
    """Calculate score based on the top N values with custom weights for each position.
    
    Args:
        vals: List of float values to score
        weights: List of weights corresponding to top N values (should sum to 1)
        
    Returns:
        Weighted average of top N values
        
    Example:
        If weights is [0.5, 0.3, 0.2], the top value gets 50% weight,
        second gets 30%, and third gets 20%
    """
    if not vals or not weights:
        return 0.0
    
    # Sort values in descending order
    sorted_vals = sorted(vals, reverse=True)
    
    # Take only as many values as we have in both lists
    n = min(len(sorted_vals), len(weights))
    
    # Only use available values and their corresponding weights
    used_vals = sorted_vals[:n]
    used_weights = weights[:n]
    
    # Normalize weights to sum to 1
    weight_sum = sum(used_weights)
    if weight_sum == 0:
        return 0.0
    
    normalized_weights = [w / weight_sum for w in used_weights]
    
    # Calculate weighted sum
    return sum(val * weight for val, weight in zip(used_vals, normalized_weights))


@validate_range
def exponential_decay_weighting(vals: list[float], decayRate: float) -> float:
    """Calculate score using exponentially decreasing weights.
    
    Args:
        vals: List of float values to score
        decayRate: Rate at which weights decrease (between 0 and 1)
        
    Returns:
        Score with exponentially decayed weights
        
    Example:
        If decayRate is 0.5, weights would be [1, 0.5, 0.25, 0.125, ...]
    """
    if not vals:
        return 0.0
    
    # Sort values in descending order
    sorted_vals = sorted(vals, reverse=True)
    
    total_weight = 0.0
    weighted_sum = 0.0
    
    for i, val in enumerate(sorted_vals):
        weight = (1 - decayRate) ** i
        weighted_sum += val * weight
        total_weight += weight
    
    # Normalize by total weight
    return weighted_sum / total_weight if total_weight > 0 else 0.0


@validate_range
def primary_plus_threshold(vals: list[float], maxWeight: float, threshold: float) -> float:
    """Calculate score based on max value plus values that exceed a threshold.
    
    Args:
        vals: List of float values to score
        maxWeight: Weight for the maximum value (between 0 and 1)
        threshold: Minimum value to include in secondary calculation
        
    Returns:
        Weighted score combining max and above-threshold values
        
    Example:
        If maxWeight is 0.7 and threshold is 5.0, max value gets 70% weight,
        and mean of values >= 5.0 (excluding max) gets 30%
    """
    if not vals:
        return 0.0
    
    max_val = max(vals)
    above_threshold = [v for v in vals if v >= threshold and v != max_val]
    
    if not above_threshold:
        return max_val
    
    mean_above_threshold = sum(above_threshold) / len(above_threshold)
    
    # Combine with weights
    return max_val * maxWeight + mean_above_threshold * (1 - maxWeight)


@validate_range
def max_plus_weighted_mean_capped(vals: list[float], meanWeights: list[float]) -> float:
    """Calculate score based on max value plus mean of weighted remaining values, capped at 1.0.
    
    This function first identifies the maximum value in the input list, then applies
    weights to the remaining values (up to the number of weights available). The final score
    is the sum of the maximum value and the mean of the weighted remaining values, capped at 1.0.
    
    Args:
        vals: List of float values to score
        meanWeights: List of weights to apply to the non-maximum values
        
    Returns:
        float: The sum of max value and mean of weighted remaining values, capped at 1.0
        
    Example:
        For vals=[0.6, 0.4, 0.3] and meanWeights=[0.7, 0.3], the max value is 0.6,
        and the mean of weighted remaining values is (0.4*0.7 + 0.3*0.3)/2 = 0.185.
        Final score is 0.6 + 0.185 = 0.785, which is below 1.0 so it stays uncapped.
    """
    if not vals:
        return 0.0
    
    # Find the maximum value
    max_val = max(vals)
    
    # If no weights or only one value, just return max_val (capped)
    if not meanWeights or len(vals) == 1:
        return min(max_val, 1.0)
    
    # Get the list of values excluding the max value
    remaining_vals = [v for v in vals if v != max_val]
    
    # Apply weights to remaining values (up to available weights)
    weighted_vals = []
    for i, val in enumerate(remaining_vals):
        if i < len(meanWeights):
            weighted_vals.append(val * meanWeights[i])
    
    # If no weighted values, return max_val (capped)
    if not weighted_vals:
        return min(max_val, 1.0)
    
    # Calculate mean of weighted values
    mean_weighted = sum(weighted_vals) / len(weighted_vals)
    
    # Return the final capped score
    return min(max_val + mean_weighted, 1.0)


"""
Example Usage of Scoring Functions

Below are examples demonstrating how to use the various scoring functions in this module.
Each example shows the input values, parameters, and expected output.
"""

def scoring_module_examples():
    """Examples demonstrating the usage of all scoring functions in this module."""
    
    # Sample data for examples
    values = [0.8, 0.6, 0.9, 0.5, 0.7]
    print(f"Sample values: {values}\n")
    
    # Example 1: max_plus_mean
    max_weight = 0.7
    result = max_plus_mean(values, max_weight)
    print(f"1. max_plus_mean (maxWeight={max_weight}):")
    print(f"   Max value is {max(values)}, mean of others is {sum([v for v in values if v != max(values)])/4}")
    print(f"   Result: {result:.4f}\n")
    
    # Example 2: max_plus_max
    max_weight = 0.8
    result = max_plus_max(values, max_weight)
    print(f"2. max_plus_max (maxWeight={max_weight}):")
    print(f"   First max: {max(values)}, second max: {max([v for v in values if v != max(values)])}")
    print(f"   Result: {result:.4f}\n")
    
    # Example 3: weighted_top_n_average
    weights = [0.5, 0.3, 0.2]
    result = weighted_top_n_average(values, weights)
    print(f"3. weighted_top_n_average (weights={weights}):")
    print(f"   Top 3 values: {sorted(values, reverse=True)[:3]}")
    print(f"   Result: {result:.4f}\n")
    
    # Example 4: exponential_decay_weighting
    decay_rate = 0.5
    result = exponential_decay_weighting(values, decay_rate)
    print(f"4. exponential_decay_weighting (decayRate={decay_rate}):")
    print(f"   Sorted values: {sorted(values, reverse=True)}")
    print(f"   Weights: [1, 0.5, 0.25, 0.125, 0.0625]")
    print(f"   Result: {result:.4f}\n")
    
    # Example 5: primary_plus_threshold
    max_weight = 0.6
    threshold = 0.7
    result = primary_plus_threshold(values, max_weight, threshold)
    print(f"5. primary_plus_threshold (maxWeight={max_weight}, threshold={threshold}):")
    print(f"   Max value: {max(values)}")
    print(f"   Values above threshold: {[v for v in values if v >= threshold and v != max(values)]}")
    print(f"   Result: {result:.4f}\n")
    
    # Example 6: max_plus_weighted_mean_capped
    mean_weights = [0.6, 0.3, 0.1]
    result = max_plus_weighted_mean_capped(values, mean_weights)
    print(f"6. max_plus_weighted_mean_capped (meanWeights={mean_weights}):")
    print(f"   Max value: {max(values)}")
    print(f"   Remaining values (up to weight count): {[v for v in values if v != max(values)][:len(mean_weights)]}")
    print(f"   Result: {result:.4f} (Capped at 1.0)")


# Uncomment to run the examples
# if __name__ == "__main__":
#     scoring_module_examples()