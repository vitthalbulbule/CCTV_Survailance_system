import cv2
import numpy as np
from ultralytics import YOLO

from src.activity.keypoint_buffer import KeypointBuffer
from src.activity.keypoint_normalizer import KeypointNormalizer
from src.activity.activity_classifier import ActivityClassifier


class PoseTracker:

    def __init__(self):

        # YOLO pose model
        self.model = YOLO("yolo26n-pose.pt")

        # Buffer stores 30 frames of keypoints for each person
        self.keypoint_buffer = KeypointBuffer(
            sequence_length=30
        )

        # Keypoint normalizer
        self.normalizer = KeypointNormalizer()

        self.classifier = ActivityClassifier()

        # Keep track of people whose sequence is already processed
        self.ready_people = set()

    def process_frame(self, frame):

        # Run pose estimation + tracking
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack_custom.yaml",
            classes=[0],
            conf=0.10,
            imgsz=960,
            verbose=False
        )

        result = results[0]

        people = []

        # Make sure detections exist
        if result.boxes is None:
            return frame, people

        # Get tracking IDs
        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
        else:
            track_ids = [None] * len(result.boxes)

        # Bounding boxes
        boxes = result.boxes.xyxy.cpu().numpy()

        # Confidence scores
        confidences = result.boxes.conf.cpu().numpy()

        # Keypoints
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()
        else:
            keypoints = [None] * len(boxes)

        # Process every detected person
        for box, track_id, confidence, pose in zip(
            boxes,
            track_ids,
            confidences,
            keypoints
        ):

            x1, y1, x2, y2 = map(int, box)

            person = {
                "id": track_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(confidence),
                "keypoints": pose.tolist()
            }

            people.append(person)

            # ------------------------------------------------
            # KEYPOINT BUFFER
            # ------------------------------------------------

            if track_id is not None:

                # Add current frame keypoints to buffer
                self.keypoint_buffer.add(
                    track_id,
                    pose.tolist()
                )

                # Check how many frames are stored
                current_length = self.keypoint_buffer.get_length(
                    track_id
                )

                # ------------------------------------------------
                # WHEN 30 FRAMES ARE READY
                # ------------------------------------------------

                if current_length == 30:

                    # Process each person only once
                    if track_id not in self.ready_people:

                        # Get sequence
                        sequence = self.keypoint_buffer.get_sequence(
                            track_id
                        )

                        print(
                            f"\nID: {track_id} | "
                            f"Sequence READY"
                        )

                        # IMPORTANT:
                        # get_sequence() returns a Python list.
                        # Convert it to NumPy array before using .shape.
                        sequence = np.array(sequence)

                        print(
                            f"Original shape: "
                            f"{sequence.shape}"
                        )

                        # ------------------------------------------------
                        # NORMALIZATION
                        # ------------------------------------------------

                        normalized_sequence = self.normalizer.normalize(
                            sequence
                        )

                        print(
                            f"Normalized shape: "
                            f"{normalized_sequence.shape}"
                        )

                        # -----------------------------------------
                        # ACTIVITY CLASSIFICATION
                        # -----------------------------------------

                        prediction, confidence = self.classifier.predict(
                            normalized_sequence
                        )

                        print(
                            f"ID: {track_id} | "
                            f"Activity: {prediction} | "
                            f"Confidence: {confidence:.2f}"
                        )

                        self.ready_people.add(track_id)

                # ------------------------------------------------
                # DRAW BOUNDING BOX
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display ID
                label = f"ID: {track_id}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Draw pose skeleton
        annotated_frame = result.plot()

        return annotated_frame, people


def main():

    video_path = "dataset/videos/fighting/fi10_xvid.mp4"

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    tracker = PoseTracker()
    frame_count = 0

    while True:

        ret, frame = cap.read()
        frame_count += 1

        if not ret:
            print("Video finished.")
            break

        frame, people = tracker.process_frame(frame)

        # Print information for current frame
        print(
            f"Frame {frame_count} | People detected: {len(people)}"
        )

        for person in people:

            print(
                f"    ID: {person['id']} | "
                f"Confidence: {person['confidence']:.2f} | "
                f"Keypoints: {len(person['keypoints'])}"
            )

        cv2.imshow(
            "CCTV - Tracking + Pose",
            frame
        )

        # Press q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()