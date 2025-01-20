import paddle
GROUP_SIZE = 4
REDUCTION_RATIO = {"max": 0.8, "min": 0.5}
def calculate_gini(sorted_scores: paddle.Tensor):
    """
    Calculate the Gini coefficient for each batch in a sorted tensor.
    """
    batch_size, n = sorted_scores.shape

    prefix_sum = paddle.cumsum(sorted_scores, axis=1)

    total_sum = prefix_sum[:, -1]

    total_diff_sum = paddle.zeros(batch_size)

    for i in range(n):
        left_contrib = sorted_scores[:, i] * i - (prefix_sum[:, i-1] if i > 0 else 0)
        right_contrib = (total_sum - prefix_sum[:, i]) - sorted_scores[:, i] * (n - i - 1)
        total_diff_sum += left_contrib + right_contrib

    gini = total_diff_sum / 2 * n * total_sum
    return gini


def update_gini_on_shift(left: paddle.Tensor, right: paddle.Tensor, left_gini: paddle.Tensor, right_gini: paddle.Tensor, new_elem: paddle.Tensor):
    """
    Incrementally update Gini coefficients for left and right parts when shifting split index to the right.
    
    Parameters:
    - left: The current left part tensor (before the shift).
    - right: The current right part tensor (before the shift).
    - left_gini: The Gini coefficient of the current left part.
    - right_gini: The Gini coefficient of the current right part.
    - new_elem: The element that will move from right to left in the shift.
    
    Returns:
    - new_left_gini: Updated Gini coefficient for the new left part.
    - new_right_gini: Updated Gini coefficient for the new right part.
    """
    _, left_size = left.shape
    _, right_size = right.shape

    left_sum = left.sum(axis=-1)
    right_sum = right.sum(axis=-1)

    new_left_sum = left_sum + new_elem
    new_left_size = left_size + 1

    new_left_total_diff = left_gini * 2 * left_size * left_sum + paddle.abs(left - new_elem).sum(axis=-1)
    new_left_gini = new_left_total_diff / (2 * new_left_size * new_left_sum + 1e-9)

    new_right_sum = right_sum - new_elem
    new_right_size = right_size - 1

    if new_right_size > 0:
        new_right_total_diff = right_gini * 2 * right_size * right_sum - paddle.abs(right - new_elem).sum(axis=-1)
        new_right_gini = new_right_total_diff / (2 * new_right_size * new_right_sum + 1e-9)
    else:
        new_right_gini = 0

    return new_left_gini, new_right_gini


def get_important_token_indices(group_attention_weights: tuple[paddle.Tensor]):
    """
    This method calculates the important token indices based on group attention weights
    """
    if group_attention_weights is None:
        raise ValueError("group_attention_weights must be provided")
    assert len(group_attention_weights) == GROUP_SIZE, f'group_attention_weights must be a tuple of length {GROUP_SIZE}'
    # tuple([batch, heads, seq_len, seq_len])
    average_attention_weights = paddle.stack(group_attention_weights).mean(axis=0) # [batch, heads, seq_len, seq)_len]
    batch_size, num_heads, seq_len, _ = average_attention_weights.shape
    importance_scores = average_attention_weights.sum(axis=-2).mean(axis=1) # [batch, seq_len]

    sorted_importance_scores = importance_scores.sort(axis=-1, descending=True)

    start, end = int(seq_len * (1 - REDUCTION_RATIO["max"])), int(seq_len * (1 - REDUCTION_RATIO["min"]))
    first_calculate = True
    max_diff = 0
    max_diff_start = 0
    while start <= end:
        h, l = sorted_importance_scores[:, :start+1], sorted_importance_scores[:, start+1:]
        if first_calculate:
            g_h, g_l = calculate_gini(h), calculate_gini(l)
            first_calculate = False
        else:
            g_h, g_l = update_gini_on_shift(h, l, g_h, g_l, sorted_importance_scores[:, start])
        diff = abs(g_h - g_l)
        if diff > max_diff:
            max_diff = diff
            max_diff_start = start
        start += 1
    threshold = sorted_importance_scores[:, max_diff_start]

    return importance_scores >= threshold