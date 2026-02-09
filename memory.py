# Memory module for storing conversation and context

class Memory:
    def __init__(self):
        self.history = []

    def add(self, entry):
        self.history.append(entry)

    def get_recent(self, n=10):
        return self.history[-n:]
