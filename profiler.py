import time

class Profiler:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.start_time = None
        self.last_time = None

    def reset(self):
        if not self.enabled:
            return
        self.start_time = time.time()
        self.last_time = self.start_time

    def log(self, label):
        if not self.enabled:
            return
        now = time.time()
        since_last = now - self.last_time
        since_start = now - self.start_time
        print(f"[Profiler] {label:20s} | +{since_last:.4f}s | Total: {since_start:.4f}s")
        self.last_time = now
