from constants import *

# calculate whether or not a box is TOO close to the camera.
def past_threshold(distance_w: float, distance_h: float, screen_cover_ratio: float, confidence: float):
    # if someone is too close (where their whole body is not even showing) we want to default to True
    if screen_cover_ratio > SCREEN_COVER_THRESHOLD: return True

    distance = (WEIGHT_H * distance_h) + ((1 - WEIGHT_H) * distance_w) # consolidate distances. Weighted
    # use confidence to reduce distance in order to stay safer on less confident predictions
    # distance = distance * (confidence) # Can make this confidence scaling exponential, or weighted in future
    print(f"Distance: {distance}")
    # if distance is too far 
    if distance < DISTANCE_THRESHOLD: return True
    return False

def estimate_human_frame(person_bbox, face_bbox):
    feet_to_center_of_head = HUMAN_HEIGHT - HUMAN_HALF_HEAD # 5 for top half of your head.
    face_center = [face_bbox[0], face_bbox[1]]
    estimated_bottom = face_center[1] - feet_to_center_of_head # Gets feet y coord: center y of the face minus the center of head to feet.
    estimated_top = face_center[1] + HUMAN_HALF_HEAD


def face_in_box_using_boxes(person_box: list, face_box: list):
    return face_in_box(
        person_box[0], 
        person_box[1], 
        person_box[2], 
        person_box[3],
        face_box[0],
        face_box[1],
        face_box[2],
        face_box[3]
    )

# p_xc is person center x axis
def face_in_box(p_xc, p_yc, p_w, p_h, f_xc, f_yc, f_w, f_h):
    person_corners = get_corners(p_xc, p_yc, p_w, p_h)
    face_corners = get_corners(f_xc, f_yc, f_w, f_h)
    
    # person sides of bounding box
    p_x_1 = person_corners[1][0] # face left side
    p_x_2 = person_corners[0][0] # face right side
    p_y_1 = person_corners[0][1] # face top side
    p_y_2 = person_corners[2][1] # face bottom side


    # face sides of bounding box
    f_x_1 = face_corners[1][0] # face left side
    f_x_2 = face_corners[0][0] # face right side
    f_y_1 = face_corners[0][1] # face top side
    f_y_2 = face_corners[2][1] # face bottom side

    diff = f_yc - p_yc # difference between centers. Good score should be posititve as head should be above body usually

    # check if any are in between
    if (
        (f_x_1 >= p_x_1 and f_x_1 <= p_x_2) or 
        (f_x_2 >= p_x_1 and f_x_2 <= p_x_2) or 
        (f_y_1 <= p_y_1 and f_y_1 >= p_y_2) or 
        (f_y_2 <= p_y_1 and f_y_2 >= p_y_2)
    ):
        return True, diff
    else:
        return False, diff
    

# returns top right, top left, bottom left, bottom right
def get_corners(center_x, center_y, width, height):
    h_width = width / 2
    h_height = height / 2

    top_right = [center_x + h_width, center_y + h_height]
    top_left = [center_x - h_width, center_y + h_height]
    bottom_left = [center_x - h_width, center_y - h_height]
    bottom_right = [center_x + h_width, center_y - h_height]

    return [top_right, top_left, bottom_left, bottom_right]
