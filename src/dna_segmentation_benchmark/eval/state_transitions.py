import numpy as np
from sklearn.metrics import confusion_matrix


def _compute_state_change_errors(
        gt_pred_arr: np.ndarray,
        label_config
):

    per_label_failures = {label_id: [] for label_id in label_config.labels.keys()}

    nuc_transitions = np.lib.stride_tricks.sliding_window_view(gt_pred_arr, (2, 2))[0]

    failled_transition_mask = mask = (nuc_transitions[:, 0, 0] == nuc_transitions[:, 1, 0]) & \
                                     (nuc_transitions[:, 0, 1] != nuc_transitions[:, 1, 1])

    failled_transitions = nuc_transitions[failled_transition_mask]

    for failed_transition in failled_transitions:

        start_nuc_id = int(failed_transition[0,0])
        gt_transition_id = int(failed_transition[0,1])
        pred_transition_id = int(failed_transition[1,1])
        per_label_failures[start_nuc_id].append((gt_transition_id, pred_transition_id))

    transition_matricies = {}
    for label_id,transition_failure_tuples in per_label_failures.items():
        if len(transition_failure_tuples) == 0:
            num_labels = len(label_config.labels)
            transition_matricies[label_id] = np.zeros((num_labels, num_labels),dtype=np.int64)
            continue

        gt_transition_ids, pred_transition_ids = zip(*transition_failure_tuples)

        transition_failure_matrix = confusion_matrix(gt_transition_ids, pred_transition_ids,labels=sorted(list(per_label_failures.keys())))
        transition_matricies[label_id] = transition_failure_matrix.astype(np.int64)

    return transition_matricies