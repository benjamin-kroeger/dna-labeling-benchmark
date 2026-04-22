def _compute_intersection_over_union_score(gt_start: int, gt_end: int, pred_start: int, pred_end: int) -> float:
    """Compute the Intersection over Union (IoU) of two bounded segments."""
    # Intersection
    i_start = max(gt_start, pred_start)
    i_end = min(gt_end, pred_end)
    intersect_len = max(0, i_end - i_start + 1)

    # Union
    u_start = min(gt_start, pred_start)
    u_end = max(gt_end, pred_end)
    union_len = u_end - u_start + 1

    iou = intersect_len / union_len if union_len > 0 else 0.0

    return iou
