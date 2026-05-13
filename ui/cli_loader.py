import time
import threading
import sys

class LoadingAnimation:
    """
    A simple command-line loading spinner animation.
    Runs in a background thread to avoid blocking main process.
    """
    def __init__(self, message="Processing"):
        self.message = message
        self.running = False
        self.thread = None
        self.states = ["|", "/", "-", "\\"]

    def _start_animation(self):
        index = 0
        while self.running:
            sys.stdout.write(f"\r{self.message} {self.states[index]}")
            sys.stdout.flush()
            index = (index + 1) % len(self.states)
            time.sleep(0.1)

    def start(self):
        """Start the loading animation."""
        self.running = True
        self.thread = threading.Thread(target=self._start_animation, daemon=True)
        self.thread.start()

    def stop(self, finish_message="Done"):
        """Stop the loading animation and show final status."""
        if self.running:
            self.running = False
            self.thread.join()
            sys.stdout.write(f"\r{self.message} {finish_message}\n")
            sys.stdout.flush()