import cv2
from ultralytics import YOLO


class PoseDetector:

    def __init__(self):
        # YOLO pose model
        self.model = YOLO("yolo26n-pose.pt")

    def detect_pose(self, frame):

        results = self.model(
            frame,
            classes=[0],
            verbose=False
        )

        result = results[0]

        # Draw pose skeleton and keypoints
        annotated_frame = result.plot()

        # Extract keypoints
        if result.keypoints is not None:

            keypoints = result.keypoints.xy.cpu().numpy()

            print("Number of people:", len(keypoints))

        return annotated_frame


def main():

    video_path = "data/input/test.mp4"

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    detector = PoseDetector()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video finished.")
            break

        # Pose estimation
        frame = detector.detect_pose(frame)

        cv2.imshow(
            "CCTV - Pose Detection",
            frame
        )

        # Press q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()