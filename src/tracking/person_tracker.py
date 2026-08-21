import cv2
from ultralytics import YOLO


class PersonTracker:

    def __init__(self):
        # Load YOLO model
        self.model = YOLO("yolo26n.pt")

    def track(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()

            for box, track_id, confidence in zip(
                boxes,
                track_ids,
                confidences
            ):

                x1, y1, x2, y2 = map(int, box)

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display ID
                label = f"ID: {track_id} | {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        return frame


def main():

    video_path = "data/input/test.mp4"

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    tracker = PersonTracker()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video finished.")
            break

        # Track people
        frame = tracker.track(frame)

        # Display
        cv2.imshow("CCTV - Person Tracking", frame)

        # Press q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()