import numpy as np
import pyzed.sl as sl
import cv2
import time
import torch
from ultralytics import YOLO
import open3d as o3d
import concurrent.futures


def get_3d_points_torch(mask_indices, depth_map, cx, cy, fx, fy, device='cuda'):
    """Convert 2D mask pixel coordinates to 3D points using depth values on GPU."""
    # Convert mask_indices to PyTorch tensors
    mask_indices = torch.tensor(mask_indices, device=device)

    # Extract u and v coordinates
    u_coords = mask_indices[:, 1]
    v_coords = mask_indices[:, 0]

    # Retrieve depth values (depth map is already on the GPU)
    depth_values = depth_map[v_coords.long(), u_coords.long()].to(device)

    # Filter out invalid depth values
    valid_mask = depth_values > 0
    u_coords = u_coords[valid_mask]
    v_coords = v_coords[valid_mask]
    depth_values = depth_values[valid_mask]

    # Convert 2D to 3D using batch computation
    x_coords = (u_coords - cx) * depth_values / fx
    y_coords = (v_coords - cy) * depth_values / fy
    z_coords = depth_values

    return torch.stack((x_coords, y_coords, z_coords), dim=-1)


def main():
    # Create a smaller coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    # Check if CUDA is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the pretrained YOLO11 model and move it to the device
    model = YOLO("yolo11m-seg.pt").to(device)

    # Create a ZED camera object
    zed = sl.Camera()

    # Set initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Set camera resolution
    init_params.camera_fps = 60  # Set camera FPS
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Enable depth sensing
    init_params.depth_minimum_distance = 0.3
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE.RIGHT_HANDED_Y_UP  # Set units to meters

    # Check if the camera is opened successfully
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera: {err}")
        exit(1)

    # Get the camera calibration parameters
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    fx, fy = calibration_params.left_cam.fx, calibration_params.left_cam.fy
    cx, cy = calibration_params.left_cam.cx, calibration_params.left_cam.cy

    # Create Mat object to hold the image
    image = sl.Mat()
    depth = sl.Mat()
    key = ''
    print("Press 'q' to quit the video feed.")

    # Initialize Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    vis.add_geometry(coordinate_frame)

    # Create named window with the ability to resize
    cv2.namedWindow("YOLO11 Segmentation+Tracking", cv2.WINDOW_NORMAL)

    fps_values = []
    frame_count = 0  # To control how frequently we update Open3D
    while key != ord('q'):
        start_time = time.time()  # Start time for FPS calculation

        # Grab an image from the camera
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)

            # Check if depth retrieval is successful
            depth_retrieval_result = zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            if depth_retrieval_result != sl.ERROR_CODE.SUCCESS:
                print(f"Error retrieving depth: {depth_retrieval_result}")
                continue

            frame = image.get_data()

            # Convert frame from 4 channels (RGBA) to 3 channels (RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Run YOLO and depth retrieval asynchronously using concurrent futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_yolo = executor.submit(model.track,
                                              source=frame,
                                              imgsz=640,
                                              max_det=20,
                                              classes = [0,39,41,64,66],
                                              half=True,
                                              persist=True,
                                              retina_masks=True,
                                              conf=0.5,
                                              device=device,
                                              tracker="ultralytics/cfg/trackers/bytetrack.yaml")

                future_depth = executor.submit(zed.retrieve_measure, depth, sl.MEASURE.DEPTH)
                results = future_yolo.result()  # Get YOLO results
                zed_depth = future_depth.result()  # Get depth map

            # Convert the ZED depth map to a NumPy array
            zed_depth_np = depth.get_data()
            if zed_depth_np is None:
                print("Error: Depth map is empty")
                continue

            # Visualize the results on the frame
            annotated_frame = results[0].plot(line_width=2, font_size=18)

            # Get the mask and bounding boxes for the detected objects
            masks = results[0].masks
            point_clouds = []

            # Process each mask
            for mask in masks.data:
                mask = mask.cpu().numpy()  # Convert mask to numpy array
                mask_indices = np.argwhere(mask > 0)  # Get indices of mask pixel

                # Convert depth map to a torch tensor for faster 3D computation
                depth_map = torch.from_numpy(zed_depth_np).to(device)

                # Use torch.cuda.amp.autocast() for mixed precision inference
                with torch.cuda.amp.autocast():
                    points_3d = get_3d_points_torch(mask_indices, depth_map, cx, cy, fx, fy)

                # Append points to list of point clouds
                if points_3d.size(0) > 0:
                    point_clouds.append(points_3d.cpu().numpy())  # Move points back to CPU

            # Visualize the 3D point clouds
            if point_clouds and frame_count % 10 == 0:  # Only update visualization every 10 frames
                # Create Open3D point cloud objects
                o3d_pcds = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc)) for pc in point_clouds]

                # Transformation matrix for 180° rotation around the x-axis
                transformation_matrix = np.array([[1, 0, 0, 0],
                                                  [0, -1, 0, 0],
                                                  [0, 0, -1, 0],
                                                  [0, 0, 0, 1]])

                # Apply the transformation to each point cloud
                for pcd in o3d_pcds:
                    pcd.transform(transformation_matrix)

                # Assign colors to each point cloud for distinction
                for pcd, color in zip(o3d_pcds, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):  # Assign different colors
                    pcd.paint_uniform_color(color)

                # Update the visualizer with new point clouds
                vis.clear_geometries()
                vis.add_geometry(coordinate_frame)
                for pcd in o3d_pcds:
                    vis.add_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

            # Increment frame count
            frame_count += 1

            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            fps_values.append(fps)

            # Calculate a moving average to smooth the FPS display
            if len(fps_values) > 10:
                fps_values.pop(0)
            avg_fps = sum(fps_values) / len(fps_values)

            # Display FPS on the frame
            cv2.putText(annotated_frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Resize the window to match the frame resolution
            height, width, _ = frame.shape
            cv2.resizeWindow("YOLO11 Segmentation+Tracking", width, height)
            cv2.imshow("YOLO11 Segmentation+Tracking", annotated_frame)

            key = cv2.waitKey(1)

    # Close the camera and exit
    zed.close()
    vis.destroy_window()


if __name__ == "__main__":
    main()