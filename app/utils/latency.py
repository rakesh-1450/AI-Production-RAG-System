import time


class LatencyTracker:

    def __init__(self):

        self.start_time = None

    def start(self):

        self.start_time = time.time()

    def stop(self):

        end_time = time.time()

        latency = end_time - self.start_time

        return round(latency, 4)