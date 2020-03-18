def get_label_dict(labels):
    label_to_idx_dict = {}
    idx_to_label_dict = {}
    idx = 0
    for label in labels:
        if label not in label_to_idx_dict.keys():
            label_to_idx_dict[label] = idx
            idx += 1

    for key,value in label_to_idx_dict.items():
        idx_to_label_dict[value] = key

    return label_to_idx_dict,idx_to_label_dict

def label_to_idx(labels,label_to_idx_dict):
    indices = []
    for label in labels:
        indices.append(label_to_idx_dict[label])
    return indices

def idx_to_label(indices, idx_to_label_dict):
    labels = []
    for index in indices:
        labels.append(idx_to_label_dict[index])
    return labels