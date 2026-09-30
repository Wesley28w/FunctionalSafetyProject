import time
from constants import *


def filter_low_conf(threshold, confidences):
    confident_results = []

    # filter the low confidence results out
    for index, conf_score in enumerate(confidences):
        if conf_score > threshold:
            confident_results.append(index)
    # return indices
    return confident_results

import time

class Debouncer:
    def __init__(self, required_time=1.0, initial_state=False):
        self.required_time = required_time
        self.current_state = initial_state
        self.last_raw_value = initial_state
        self.transition_time = None

    def touch(self, value: bool) -> bool:
        current_time = time.time()

        # If the input value changed from the last time we checked
        if value != self.last_raw_value:
            self.transition_time = current_time
            self.last_raw_value = value

        # If the value is different from our stable output state
        if value != self.current_state:
            # Check if it has held this new value long enough
            if self.transition_time and (current_time - self.transition_time >= self.required_time):
                self.current_state = value  # Lock in the new stable state

        return self.current_state
