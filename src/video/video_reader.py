import cv2


VIDEO_PATH = "data/input/test.mp4"


def read_video():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Video Information")
    print("------------------")
    print("FPS:", fps)
    print("Width:", width)
    print("Height:", height)

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video finished.")
            break

        cv2.imshow("CCTV Video", frame)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    read_video()