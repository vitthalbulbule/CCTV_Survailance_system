import os
import csv
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

from src.activity.keypoint_normalizer import KeypointNormalizer


VIDEO_DIR = "dataset/videos"
SEQUENCE_DIR = "dataset/sequence"

SEQUENCE_LENGTH = 30

# Sliding window
STRIDE = 5

# Allow very small tracking/detection gaps
MAX_GAP = 2

MODEL_PATH = "yolo26n-pose.pt"
TRACKER_PATH = "bytetrack_custom.yaml"

CLASSES = [
    "fighting",
    "non_fighting"
]

MAX_VIDEOS_PER_CLASS = None


class SequenceGenerator:

    def __init__(self):
        self.normalizer = KeypointNormalizer()

    def interpolate_pose(self, previous_pose, current_pose, gap):

        previous_pose = np.array(
            previous_pose,
            dtype=np.float32
        )

        current_pose = np.array(
            current_pose,
            dtype=np.float32
        )

        interpolated = []

        for i in range(1, gap + 1):

            alpha = i / (gap + 1)

            pose = (
                previous_pose
                + alpha * (current_pose - previous_pose)
            )

            interpolated.append(
                pose.tolist()
            )

        return interpolated

    def save_sequence(
        self,
        buffer,
        output_dir,
        video_name,
        track_id,
        sequence_number
    ):

        sequence = np.array(
            buffer,
            dtype=np.float32
        )

        if sequence.shape != (30, 17, 2):
            return False

        normalized_sequence = (
            self.normalizer.normalize(sequence)
        )

        output_name = (
            f"{video_name}"
            f"_person_{track_id}"
            f"_seq_{sequence_number:03d}.npy"
        )

        output_path = os.path.join(
            output_dir,
            output_name
        )

        np.save(
            output_path,
            normalized_sequence
        )

        print(
            f"Saved: {output_name} "
            f"| Shape: {normalized_sequence.shape}"
        )

        return True

    def process_video(self, video_path, output_dir):

        print("\n" + "=" * 60)
        print(f"Processing: {video_path}")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"ERROR: Could not open {video_path}")
            return 0

        # Fresh model/tracker for every video
        model = YOLO(MODEL_PATH)

        # Each person has its own 30-frame buffer
        buffers = defaultdict(
            lambda: deque(maxlen=SEQUENCE_LENGTH)
        )

        # Last pose detected for each ID
        last_pose = {}

        # Last frame number for each ID
        last_seen = {}

        # Sequence counter for this video
        sequence_counter = 0

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_PATH,
                classes=[0],

                # Lower detection threshold
                conf=0.2,

                verbose=False
            )

            result = results[0]

            if result.boxes is None:
                continue

            if result.boxes.id is None:
                continue

            if result.keypoints is None:
                continue

            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            keypoints = (
                result.keypoints.xy
                .cpu()
                .numpy()
            )

            for track_id, pose in zip(
                track_ids,
                keypoints
            ):

                if pose.shape != (17, 2):
                    continue

                # ==================================================
                # CHECK TRACK CONTINUITY
                # ==================================================

                if track_id in last_seen:

                    gap = (
                        frame_count
                        - last_seen[track_id]
                        - 1
                    )

                    # ------------------------------------------------
                    # Small gap:
                    # interpolate missing frames
                    # ------------------------------------------------

                    if 0 < gap <= MAX_GAP:

                        missing_poses = (
                            self.interpolate_pose(
                                last_pose[track_id],
                                pose,
                                gap
                            )
                        )

                        for missing_pose in missing_poses:
                            buffers[track_id].append(
                                missing_pose
                            )

                    # ------------------------------------------------
                    # Large gap:
                    # Do NOT combine unrelated movements
                    # ------------------------------------------------

                    elif gap > MAX_GAP:

                        buffers[track_id].clear()

                # Add current pose
                buffers[track_id].append(
                    pose.tolist()
                )

                last_pose[track_id] = pose.tolist()
                last_seen[track_id] = frame_count

                # ==================================================
                # 30 FRAME SEQUENCE READY
                # ==================================================

                if len(buffers[track_id]) == SEQUENCE_LENGTH:

                    video_name = os.path.splitext(
                        os.path.basename(video_path)
                    )[0]

                    saved = self.save_sequence(
                        buffers[track_id],
                        output_dir,
                        video_name,
                        track_id,
                        sequence_counter
                    )

                    if saved:

                        sequence_counter += 1

                        # Sliding window
                        for _ in range(STRIDE):

                            if buffers[track_id]:
                                buffers[track_id].popleft()

        cap.release()

        print("\nFinished:")
        print(f"Video       : {video_path}")
        print(f"Frames      : {frame_count}")
        print(f"Sequences   : {sequence_counter}")

        return sequence_counter

    def process_dataset(self):

        summary = []

        for class_name in CLASSES:

            input_dir = os.path.join(
                VIDEO_DIR,
                class_name
            )

            output_dir = os.path.join(
                SEQUENCE_DIR,
                class_name
            )

            os.makedirs(output_dir, exist_ok=True)

            if not os.path.exists(input_dir):

                print(
                    f"Directory not found: "
                    f"{input_dir}"
                )

                continue

            videos = sorted([
                file
                for file in os.listdir(input_dir)
                if file.lower().endswith(
                    (".mp4", ".avi", ".mov", ".mkv")
                )
            ])

            if MAX_VIDEOS_PER_CLASS is not None:

                videos = videos[
                    :MAX_VIDEOS_PER_CLASS
                ]

            print("\n" + "#" * 60)

            print(
                f"Class: {class_name} "
                f"| Videos: {len(videos)}"
            )

            print("#" * 60)

            class_total = 0

            for video in videos:

                video_path = os.path.join(
                    input_dir,
                    video
                )

                count = self.process_video(
                    video_path,
                    output_dir
                )

                class_total += count

                summary.append({
                    "class": class_name,
                    "video": video,
                    "sequences": count
                })

            print(
                f"\nTOTAL {class_name.upper()} "
                f"SEQUENCES: {class_total}"
            )

        # ==========================================================
        # SAVE SUMMARY
        # ==========================================================

        summary_path = os.path.join(
            SEQUENCE_DIR,
            "sequence_summary.csv"
        )

        with open(
            summary_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "class",
                    "video",
                    "sequences"
                ]
            )

            writer.writeheader()
            writer.writerows(summary)

        print("\n" + "=" * 60)
        print("DATASET PROCESSING COMPLETE")
        print("=" * 60)

        print(
            f"Summary saved to: {summary_path}"
        )


if __name__ == "__main__":

    generator = SequenceGenerator()

    generator.process_dataset()