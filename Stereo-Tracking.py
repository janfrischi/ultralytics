import pyzed.sl as sl
import cv2
import time
import torch
from ultralytics import YOLO

def main():
    # Check if CUDA is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the YOLO11 model
    model = YOLO("yolo11m-seg.pt").to(device)

    # Create a ZED camera object
    zed = sl.Camera()

    # Set initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Set camera resolution
    init_params.camera_fps = 60  # Set camera FPS
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # Disable depth for now, we only need RGB images
    init_params.coordinate_units = sl.UNIT.METER  # Set units to meters

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera: {err}")
        exit(1)

    # Create Mat objects to hold the left and right images
    left_image = sl.Mat()
    right_image = sl.Mat()

    key = ''
    print("Press 'q' to quit the video feed.")

    while key != ord('q'):
        start_time = time.time()  # Start time for FPS calculation

        # Grab an image from the camera
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve both left and right images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)

            # Convert left and right frames from 4 channels (RGBA) to 3 channels (BGR)
            left_frame = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2BGR)
            right_frame = cv2.cvtColor(right_image.get_data(), cv2.COLOR_BGRA2BGR)

            # Perform object tracking on the left image
            results_left = model.track(
                source=left_frame, imgsz=640, max_det=20, classes=[41,64,66], conf=0.5,
                device=device, retina_masks=True, persist=True, tracker="ultralytics/cfg/trackers/botsort.yaml"
            )
            annotated_left_frame = results_left[0].plot(labels=True, line_width=2, font_size=18)

            # Perform object tracking on the right image
            results_right = model.track(
                source=right_frame, imgsz=640, max_det=20, classes=[41,64,66], conf=0.5,
                device=device, retina_masks=True, persist=True, tracker="ultralytics/cfg/trackers/botsort.yaml"
            )
            annotated_right_frame = results_right[0].plot(labels=True, line_width=2, font_size=18)

            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)

            # Annotate FPS on both images
            cv2.putText(annotated_left_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_right_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the left and right annotated frames
            cv2.imshow("YOLO11 Segmentation+Tracking Left Image", annotated_left_frame)
            cv2.imshow("YOLO11 Segmentation+Tracking Right Image", annotated_right_frame)

        # Wait for key press
        key = cv2.waitKey(10)

    # Close the camera and exit
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
