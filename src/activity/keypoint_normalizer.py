import numpy as np


class KeypointNormalizer:

    def __init__(self):
        pass

    def normalize(self, sequence):
        """
        Normalize a sequence of human pose keypoints.

        Input:
            sequence shape = (30, 17, 2)

        Output:
            normalized sequence shape = (30, 17, 2)
        """

        sequence = np.array(sequence, dtype=np.float32)

        # Check expected dimensions
        if sequence.ndim != 3:
            raise ValueError(
                f"Expected 3 dimensions, got {sequence.ndim}"
            )

        # Get number of frames, keypoints and coordinates
        frames, keypoints, coordinates = sequence.shape

        print(
            f"Input shape: {sequence.shape}"
        )

        # We expect:
        # 30 frames
        # 17 keypoints
        # 2 coordinates (x, y)

        if keypoints != 17 or coordinates != 2:
            raise ValueError(
                f"Expected shape (frames, 17, 2), "
                f"got {sequence.shape}"
            )

        normalized_sequence = np.zeros_like(sequence)

        for frame in range(frames):

            # COCO keypoint indices:
            # 11 = left hip
            # 12 = right hip

            left_hip = sequence[frame, 11]
            right_hip = sequence[frame, 12]

            # Calculate center of both hips
            hip_center = (left_hip + right_hip) / 2

            # Move all keypoints relative to hip center
            normalized_sequence[frame] = (
                sequence[frame] - hip_center
            )

        print(
            f"Output shape: {normalized_sequence.shape}"
        )

        return normalized_sequence