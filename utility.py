from constants import *

def filter_low_conf(threshold, confidences):
    confident_results = []

    # filter the low confidence results out
    for index, conf_score in enumerate(confidences):
        if conf_score > threshold:
            confident_results.append(index)
    # return indices
    return confident_results