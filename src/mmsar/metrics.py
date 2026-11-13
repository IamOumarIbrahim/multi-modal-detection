#   MMSAR - frame and alert metrics
#        Oumar Ibrahim U22200741


# precision recall and f1 over binary frame labels
def calculate_frame_metrics(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different length")

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn += 1

    precision = 0.0
    if tp + fp > 0:
        precision = tp / (tp + fp)
    recall = 0.0
    if tp + fn > 0:
        recall = tp / (tp + fn)
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


# returns [start, end) of every run of 1s
def extract_events(seq):
    events = []
    in_event = False
    start = 0
    for i in range(len(seq)):
        if seq[i] == 1 and not in_event:
            in_event = True
            start = i
        elif seq[i] == 0 and in_event:
            in_event = False
            events.append((start, i))
    if in_event:
        events.append((start, len(seq)))
    return events


# false alerts = alert runs on frames with no target
def count_false_alerts(y_true, y_pred):
    masked = []
    for i in range(len(y_pred)):
        if y_true[i] == 0:
            masked.append(y_pred[i])
        else:
            masked.append(0)
    return len(extract_events(masked))


# frames from target onset to the first alert, for every target event that got alerted
def time_to_alarm(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different length")

    delays = []
    for start, end in extract_events(y_true):
        for i in range(start, end):
            if y_pred[i] == 1:
                delays.append(i - start)
                break
    return delays
