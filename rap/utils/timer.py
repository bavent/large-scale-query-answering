"""Tiny stopwatch-esque timer, with start/stop functionality."""

import time


class Timer:
    """Computes total elapsed time."""

    def __init__(self):
        """Initializes new Timer object and starts timing."""
        self.total_time = 0.
        self.is_running = False
        self.start()

    def stop(self):
        """Stops the timer from accumulating time."""
        if self.is_running:
            self.total_time += (time.time() - self.start_time)
            self.is_running = False

    def start(self):
        """Starts the timer accumulating time."""
        if not self.is_running:
            self.start_time = time.time()
            self.is_running = True

    def get_total_time(self) -> float:
        """Retrieves how long the timer has been running for.

        Returns:
            float: Total number of seconds the timer has been running for.
        """
        if self.is_running:
            self.stop()
            total_time = self.total_time
            self.start()
        else:
            total_time = self.total_time
        return total_time

    def add_to_total_time(self, amount_to_add: float):
        """Adds a given amount of time to the Timer's tracked total time.

        Args:
            amount_to_add (float): Amount of time to add to Timer.
        """
        self.total_time += amount_to_add
