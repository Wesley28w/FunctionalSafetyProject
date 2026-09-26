import cv2
import math

# CONSTANTS:
FOV_H = 122.0 # how wide is the camera. (Currently for Mac Air M4)
RESOLUTION = (1920.0, 1080.0) # pixel size resolution
STREAM_W = RESOLUTION[0]
STREAM_H = RESOLUTION[1]
FOCAL_LENGTH = STREAM_W / (2 * math.tan(FOV_H / 2))
FOV_V = 2 * math.atan(STREAM_H / (2 * FOCAL_LENGTH)) # focal length is same for V/H FOV

# TODO: make this work for kids too (maybe use ratio of both?)
# LIMITATION: if someone is too close to camera where you can't see entire body then distance will be off. But should disable regardless.
HUMAN_HEIGHT = 68.0 # inches - 5'8"
HUMAN_WIDTH = 19.0 # inches - Shoulder SPAN
ASPECT_RATIO = HUMAN_HEIGHT / HUMAN_WIDTH

# Calculation Constants
# CURRENTLY NOT USING WIDTH BECAUSE VARIES TO HEAVILY
WEIGHT_H = 1.0 # How much to weigh height distance calculation over width
SCREEN_COVER_THRESHOLD = 0.3 # How much of the screen a person covers to be considered too close
DISTANCE_THRESHOLD = 32 # inches. 10 feet

# for text display
org = (50, 50)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1 
thickeness = 2
line_type = cv2.LINE_AA