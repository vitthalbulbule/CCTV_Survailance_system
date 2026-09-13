from collections import defaultdict, deque


class KeypointBuffer:

    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length

        self.buffers = defaultdict(
            lambda: deque(maxlen=sequence_length)
        )

    def add(self, person_id, keypoints):
        self.buffers[person_id].append(keypoints)

    def is_ready(self, person_id):
        return len(self.buffers[person_id]) == self.sequence_length

    def get_sequence(self, person_id):
        if not self.is_ready(person_id):
            return None

        return list(self.buffers[person_id])

    def remove_person(self, person_id):
        if person_id in self.buffers:
            del self.buffers[person_id]

    def get_length(self, person_id):
        return len(self.buffers[person_id])