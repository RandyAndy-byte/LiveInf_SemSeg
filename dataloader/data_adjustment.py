import numpy as np
from random import random
import yaml

def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    return label - 1  # uint8 trick


def train2SemKITTI(input_label):
    # delete 0 label (uses uint8 trick : 0 - 1 = 255 )
    return input_label + 1

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist


def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]
    nusc_learning = nuScenesyaml['learning_map']
    return nuScenes_label_name, nusc_learning

# Save output with removed
def createFiles(arr, remove=False):
    filename = "visualizations/" + str(random.randint(0, 99999))
    np.savetxt(filename + ".txt", arr, fmt="%s")

    if remove is True:
        removed_arr = removeObject(arr, "people")
        filename += "removed"
        np.savetxt(filename + ".txt", removed_arr, fmt="%s")
    # filename += "two"
    # np.savetxt(filename + ".txt", clean_arr, fmt="%s")


# Assign color values to labels via mapping from the YAML config file
def genColors(pred, learning_map, color_map): # Call label map for labels in arguements
    translated_numbers = np.array([learning_map[number] for number in pred])
    #translated_labels = np.array([label_map[label] for label in translated_numbers])
    translated_colors = np.array([color_map[colors] for colors in translated_numbers])
    #translated_labels = np.expand_dims(translated_labels, axis=0).T
    #return np.hstack((translated_labels, translated_colors))
    return translated_colors/255

# With labels
def genColors(pred, learning_map, color_map, label_map): # Call label map for labels in arguements
    translated_numbers = np.array([learning_map[number] for number in pred])
    translated_labels = np.array([label_map[label] for label in translated_numbers])
    translated_colors = np.array([color_map[colors] for colors in translated_numbers])/255
    translated_labels = np.expand_dims(translated_labels, axis=0).T
    return np.hstack((translated_labels, translated_colors))

# Assign color values to labels via mapping from the YAML config file
def genColorsNusc(pred, color_map): # Call label map for labels in arguements
    #translated_numbers = np.array([learning_map[number] for number in pred])
    #translated_labels = np.array([label_map[label] for label in translated_numbers])
    translated_colors = np.array([color_map[colors] for colors in pred])
    #translated_labels = np.expand_dims(translated_labels, axis=0).T
    #return np.hstack((translated_labels, translated_colors))
    return translated_colors/255

def labels2colors(labels, color_map):
    colors = np.array(color_map[point] for point in labels)
    print(labels)
    print(colors)

# Remove specified object from the lidar scan
def removeObject(arr, obj):
    removed_arr = np.delete(arr, np.where(arr[:, 4] == obj), axis=0)
    removed_arr = np.delete(removed_arr, 4, axis=1)
    # clean_arr = np.delete(arr, np.where(arr[:, 3] != "person"), axis=0)
    return removed_arr.astype(np.float32)  # , clean_arr