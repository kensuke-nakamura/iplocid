"""POIL metrics for evaluated query records (no model dependencies)."""
import math


def poil_metrics(outputs, threshold=0.5):
    """Use existing PN labels; only positive success requires IoU >= threshold."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('IoU threshold must be in [0, 1]')
    p = n = s_pos = s_neg = 0
    for record in outputs:
        role = record['role']
        if role not in ('positive-image', 'negative-image', 'inclass-image', 'outclass-image'):
            raise ValueError('Unsupported query role: ' + str(role))
        label = record['pn_label']
        if label not in ('positive', 'negative'):
            raise ValueError('Unsupported response: ' + str(label))
        if role == 'positive-image':
            p += 1
            value = record.get('iou_raw')
            iou = float(value) if value is not None else 0.0
            if not math.isfinite(iou):
                iou = 0.0
            s_pos += int(label == 'positive' and iou >= threshold)
        else:
            n += 1
            s_neg += int(label == 'negative')
    tpr = s_pos / p if p else None
    tnr = s_neg / n if n else None
    accuracy = (tpr + tnr) / 2 if p and n else None
    score = None
    if p and n:
        score = 2 * tpr * tnr / (tpr + tnr) if tpr + tnr else 0.0
    return {'P': p, 'N': n, 'S_pos': s_pos, 'S_neg': s_neg,
            'iou_threshold': threshold, 'joint_TPR': tpr, 'TNR': tnr,
            'POIL_Acc': accuracy, 'POIL_Score': score}
