import cv2
from ultralytics import YOLO


class PersonDetector:

    def __init__(self):
        # Load pretrained YOLO model
        self.model = YOLO("yolo26n.pt")

    def detect(self, frame):

        # Run YOLO on the current frame
        results = self.model(frame, verbose=False)

        for result in results:

            boxes = result.boxes

            for box in boxes:

                # Class ID
                class_id = int(box.cls[0])

                # Confidence
                confidence = float(box.conf[0])

                # YOLO COCO class 0 = person
                if class_id == 0:

                    # Bounding box coordinates
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Label
                    label = f"Person {confidence:.2f}"

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

    # Load video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create detector
    detector = PersonDetector()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video finished.")
            break

        # Detect people
        frame = detector.detect(frame)

        # Display
        cv2.imshow("CCTV - Person Detection", frame)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()