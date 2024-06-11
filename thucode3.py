import cv2
import numpy as np
from ultralytics import YOLO, solutions
from tracker import *
import pandas as pd
import os
import sys

from collections import defaultdict
from time import time
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors


traffic_record_folder_name = "TrafficRecord"
if not os.path.exists(traffic_record_folder_name):
    os.makedirs(traffic_record_folder_name)
    os.makedirs(traffic_record_folder_name+"//exceeded")
speed_record_file_location = traffic_record_folder_name + "//SpeedRecord.txt"
global filet
filet = open(speed_record_file_location, "w")
# W là write
filet.write("ID \t SPEED\n------\t-------\n")
filet.close() 


class SpeedEstimator:
    """A class to estimate the speed of objects in a real-time video stream based on their tracks."""

    def __init__(self, names, reg_pts=None, view_img=False, line_thickness=2, region_thickness=5, spdl_dist_thresh=10):
        """
        Initializes the SpeedEstimator with the given parameters.

        Args:
            names (dict): Dictionary of class names.
            reg_pts (list, optional): List of region points for speed estimation. Defaults to [(20, 400), (1260, 400)].
            view_img (bool, optional): Whether to display the image with annotations. Defaults to False.
            line_thickness (int, optional): Thickness of the lines for drawing boxes and tracks. Defaults to 2.
            region_thickness (int, optional): Thickness of the region lines. Defaults to 5.
            spdl_dist_thresh (int, optional): Distance threshold for speed calculation. Defaults to 10.
        """
        # Visual & image information
        self.im0 = None
        self.annotator = None
        self.view_img = view_img

        # Region information
        self.reg_pts = reg_pts if reg_pts is not None else [(20, 400), (1260, 400)]
        self.region_thickness = region_thickness

        # Tracking information
        self.clss = None
        self.names = names
        self.boxes = None
        self.trk_ids = None
        self.trk_pts = None
        self.line_thickness = line_thickness
        self.trk_history = defaultdict(list)
        self.boxlist= []

        # Speed estimation information
        self.current_time = 0
        self.dist_data = {}
        self.trk_idslist = []
        self.spdl_dist_thresh = spdl_dist_thresh
        self.trk_previous_times = {}
        self.trk_previous_points = {}
        self.tocdo = 0 
        self.exceeded = 0

        # Check if the environment supports imshow
        self.env_check = check_imshow(warn=True)

    def extract_tracks(self, tracks):
        """
        Extracts results from the provided tracking data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()

    def store_track_info(self, track_id, box):
        """
        Stores track data.

        Args:
            track_id (int): Object track id.
            box (list): Object bounding box data.

        Returns:
            (list): Updated tracking history for the given track_id.
        """
        track = self.trk_history[track_id]
        bbox_center = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
        track.append(bbox_center)

        if len(track) > 30:
            track.pop(0)

        self.trk_pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        return track

    def plot_box_and_track(self, track_id, box, cls, track):
        """
        Plots track and bounding box.

        Args:
            track_id (int): Object track id.
            box (list): Object bounding box data.
            cls (str): Object class name.
            track (list): Tracking history for drawing tracks path.
        """
        speed_label = f"{int(self.dist_data[track_id])} km/h" if track_id in self.dist_data else self.names[int(cls)]
        bbox_color = colors(int(track_id)) if track_id in self.dist_data else (255, 0, 255)

        self.annotator.box_label(box, speed_label, bbox_color)
        cv2.polylines(self.im0, [self.trk_pts], isClosed=False, color=(0, 255, 0), thickness=1)
        cv2.circle(self.im0, (int(track[-1][0]), int(track[-1][1])), 5, bbox_color, -1)

    def calculate_speed(self, trk_id, track):
        """
        Calculates the speed of an object.

        Args:
            trk_id (int): Object track id.
            track (list): Tracking history for drawing tracks path.
        """
        if not self.reg_pts[0][0] < track[-1][0] < self.reg_pts[1][0]:
            return
        if self.reg_pts[1][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[1][1] + self.spdl_dist_thresh:
            direction = "known"
        elif self.reg_pts[0][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[0][1] + self.spdl_dist_thresh:
            direction = "known"
        else:
            direction = "unknown"

        if self.trk_previous_times.get(trk_id) != 0 and direction != "unknown" and trk_id not in self.trk_idslist:
            self.trk_idslist.append(trk_id)

            time_difference = time() - self.trk_previous_times[trk_id]
            if time_difference > 0:
                dist_difference = np.abs(track[-1][1] - self.trk_previous_points[trk_id][1])
                speed = dist_difference / time_difference*6
                self.dist_data[trk_id] = speed
                self.tocdo = speed

        self.trk_previous_times[trk_id] = time()
        self.trk_previous_points[trk_id] = track[-1]

    def capture(self, im0, x, y, h, w, sp, id):
        xmi=int(x)
        ymi=int(y)
        xma=int(w)
        yma=int(h)
        if (sp > 70):
            crop_img = im0[ymi - 5:ymi + yma + 5, xmi - 5:xmi + xma + 5]
            n = str(id) + "_speed_" + str(sp)
            file = traffic_record_folder_name + '//exceeded//' + n + '.jpg'
            cv2.imwrite(file, crop_img)
            # filet = open(speed_record_file_location, "a")
            # append là ghi giá trị ở cuối file 
            
            # filet.write(str(id) + " \t " + str(sp) + "<---exceeded\n")
            self.exceeded += 1
        # filet.close()


    def estimate_speed(self, im0, tracks, region_color=(255, 0, 0)):
        """
        Estimates the speed of objects based on tracking data.
        
        Args:
            im0 (ndarray): Image.
            tracks (list): List of tracks obtained from the object tracking process.
            region_color (tuple, optional): Color to use when drawing regions. Defaults to (255, 0, 0).

        Returns:
            (ndarray): The image with annotated boxes and tracks.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            if self.view_img and self.env_check:
                self.display_frames()
            return im0

        self.extract_tracks(tracks)
        self.annotator = Annotator(self.im0, line_width=self.line_thickness)
        self.annotator.draw_region(reg_pts=self.reg_pts, color=region_color, thickness=self.region_thickness)

        for box, trk_id, cls in zip(self.boxes, self.trk_ids, self.clss):
            track = self.store_track_info(trk_id, box)

            if trk_id not in self.trk_previous_times:
                self.trk_previous_times[trk_id] = 0

            self.plot_box_and_track(trk_id, box, cls, track)
            self.calculate_speed(trk_id, track)

            self.boxlist = list(box)

            


        if self.view_img and self.env_check:
            self.display_frames()

        return im0

    def display_frames(self):
        """Displays the current frame."""
        cv2.imshow("Ultralytics Speed Estimation", self.im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


# Load the best fine-tuned YOLOv8 model
model = YOLO(r'D:\hocphan\XLA\YOLOv8_Traffic_Density_Estimation-master\models\best.pt')
model.export(format="onnx")
names = model.model.names

# Define the threshold for considering traffic as heavy
heavy_traffic_threshold = 2

# Define the vertices for the quadrilaterals
vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

# Define the vertical range for the slice and lane threshold
x1, x2 = 300, 655 
lane_threshold = 609

# Define the positions for the text annotations on the image
text_position_left_lane = (10, 50)
text_position_right_lane = (820, 50)
intensity_position_left_lane = (10, 100)
intensity_position_right_lane = (820, 100)


# Define font, scale, and colors for the annotations
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)    # White color for text
background_color = (0, 0, 255)  # Red background for text

# Open the video
cap = cv2.VideoCapture('sample_video.mp4')

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

line_pts = [(0, 440), (1280, 440)]

# Init speed-estimation obj
speed_obj = SpeedEstimator(reg_pts=line_pts, names=names, view_img=False,)
# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Create a copy of the original frame to modify
        detection_frame = frame.copy()
    
        # Black out the regions outside the specified vertical range
        detection_frame[:x1, :] = 0  # Black out from top to x1
        detection_frame[x2:, :] = 0  # Black out from x2 to the bottom of the frame
        
        # Perform inference on the modified frame
        results = model.predict(detection_frame, imgsz=640, conf=0.4)
        processed_frame = results[0].plot(line_width=1)
        
        # Restore the original top and bottom parts of the frame
        processed_frame[:x1, :] = frame[:x1, :].copy()
        processed_frame[x2:, :] = frame[x2:, :].copy()        
        
        # Draw the quadrilaterals on the processed frame
        cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(processed_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)
        
        #SPEED
        ##############################################################
        tracks = model.track(detection_frame, persist=True, show=False)

        frame = speed_obj.estimate_speed(processed_frame, tracks)
       
        
        video_writer.write(processed_frame)
        ##############################################################

        #CAPTUREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        #MÀ chưa biết bounding boxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        for id, sp in speed_obj.dist_data.items():
            l= speed_obj.boxlist
            x_min = l[0]
            y_min = l[1]
            x_max = l[2]
            y_max = l[3]

            speed_obj.capture(processed_frame, x_min, y_min, y_max, x_max, speed_obj.dist_data[id], id)
       
        # Retrieve the bounding boxes from the results
        bounding_boxes = results[0].boxes

        

        vehicles_in_left_lane = 0
        vehicles_in_right_lane = 0

        # Loop through each bounding box to count vehicles in each lane
        for box in bounding_boxes.xyxy:
            # Check if the vehicle is in the left lane based on the x-coordinate of the bounding box
            if box[0] < lane_threshold:
                vehicles_in_left_lane += 1
            else:
                vehicles_in_right_lane += 1
            


        traffic_intensity_left = "Smooth"
        traffic_intensity_right = "Smooth"
        for id, sp in speed_obj.dist_data.items():        
            # Determine the traffic intensity for the left lane
            traffic_intensity_left = "Heavy" if ((vehicles_in_left_lane > heavy_traffic_threshold)&(speed_obj.dist_data[id] < 50)) else "Smooth"
            # Determine the traffic intensity for the right lane
            traffic_intensity_right = "Heavy" if ((vehicles_in_right_lane > heavy_traffic_threshold)&(speed_obj.dist_data[id] < 50)) else "Smooth"


        # Add a background rectangle for the left lane vehicle count
        cv2.rectangle(processed_frame, (text_position_left_lane[0]-10, text_position_left_lane[1] - 25), 
                      (text_position_left_lane[0] + 460, text_position_left_lane[1] + 10), background_color, -1)

        # Add the vehicle count text on top of the rectangle for the left lane
        cv2.putText(processed_frame, f'Vehicles in Left Lane: {vehicles_in_left_lane}', text_position_left_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Add a background rectangle for the left lane traffic intensity
        cv2.rectangle(processed_frame, (intensity_position_left_lane[0]-10, intensity_position_left_lane[1] - 25), 
                      (intensity_position_left_lane[0] + 460, intensity_position_left_lane[1] + 10), background_color, -1)

        # Add the traffic intensity text on top of the rectangle for the left lane
        cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_left}', intensity_position_left_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Add a background rectangle for the right lane vehicle count
        cv2.rectangle(processed_frame, (text_position_right_lane[0]-10, text_position_right_lane[1] - 25), 
                      (text_position_right_lane[0] + 460, text_position_right_lane[1] + 10), background_color, -1)

        # Add the vehicle count text on top of the rectangle for the right lane
        cv2.putText(processed_frame, f'Vehicles in Right Lane: {vehicles_in_right_lane}', text_position_right_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Add a background rectangle for the right lane traffic intensity
        cv2.rectangle(processed_frame, (intensity_position_right_lane[0]-10, intensity_position_right_lane[1] - 25), 
                      (intensity_position_right_lane[0] + 460, intensity_position_right_lane[1] + 10), background_color, -1)

        # Add the traffic intensity text on top of the rectangle for the right lane
        cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_right}', intensity_position_right_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Display the processed frame
        cv2.imshow('Real-time Traffic Analysis', processed_frame)

        # Press Q on keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
video_writer.release()
cv2.destroyAllWindows()

