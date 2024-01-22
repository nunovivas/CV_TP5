import random
import cv2 as cv2
from statistics import mean as mean
import time
import math

import numpy as np


# Store fps values for calculating mean
fps_list = []

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def center_distance(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    center1 = (x1 + w1 / 2, y1 + h1 / 2)
    center2 = (x2 + w2 / 2, y2 + h2 / 2)

    distance = math.sqrt((center2[0] - center1[0]) ** 2 + (center2[1] - center1[1]) ** 2)

    return distance   

def doMask(frame, box, newImage):
    try:
        # Load two images
        img1 = frame
        img2 = newImage
        newImage = cv2.resize(newImage, (box[2], box[3]))
        assert img2 is not None, "file could not be read, check with os.path.exists()"
        # instead create it where the box is
        roi = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        
        #Not needed
        # Now create a mask of logo and create its inverse mask also
        #img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        # old mask from 10 to 255
        #ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        #new mask 
        # Threshold the alpha channel to create a binary mask
        alpha_channel = img2[:, :, 3]
        ret, mask = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)

        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of logo in ROI
        # Check for empty images or masks
        if roi is not None and mask_inv is not None:
            # Rebenta nesta linha no cesto. Na bola não
            print("OLD")
            print(roi.dtype, mask_inv.dtype, img2.dtype,mask.dtype)
            print(roi.shape, mask_inv.shape, img2.shape,mask.shape)
            roi = roi.astype(np.int8)
            mask_inv = mask_inv.astype(np.int8)
            img2 = img2.astype(np.int8)
            mask = mask.astype(np.int8)
            print("NEW")
            print(roi.dtype, mask_inv.dtype, img2.dtype,mask.dtype)
            print(roi.shape, mask_inv.shape, img2.shape,mask.shape)

            # o problema está aqui
            img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
        # Put logo in ROI and modify the main image
        
        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg,img2_fg[:,:,:3])
        #frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :3] = img2_fg[:, :, :3]
        frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2], :3] = dst


        return True
        print ("what the fuck")
    except Exception as e:
        # Handle other types of exceptions
        print(f"Do mask: An unexpected error occurred: {e}")
        return False
   
def doMaskOld(frame, box, newImage):

    # Resize the image to fit the bounding box
    newImage_resized = cv2.resize(newImage, (box[2], box[3]))
    # Print shapes for troubleshooting
    # print("Image resized shape:", newImage_resized.shape)
    # print("Box:", box)
    # print("Frame shape:", frame.shape[0], frame.shape[1])

    # check if it has alpha channel
    if (not has_alpha_opencv(newImage)):
        print("No alpha channel on the image")
        return 0
    # Validate the box coordinates
    if (
            0 <= box[0] < frame.shape[1] and
            0 <= box[1] < frame.shape[0] and
            0 <= box[0] + box[2] <= frame.shape[1] and
            0 <= box[1] + box[3] <= frame.shape[0]
    ):
        # Create a mask based on the alpha channel (transparency)
        alpha_channel = newImage_resized[:, :, 3] / 255.0

        # Extract the region of interest from the frame
        roi = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

        # Blend the images using the alpha channel
        for c in range(0, 3):
            roi[:, :, c] = (alpha_channel * newImage_resized[:, :, c] +
                            (1 - alpha_channel) * roi[:, :, c])

        # Update the frame with the blended result
        frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = roi
        return True
    else:
        print("Invalid box size")
        return False
#delete this
#et, mask = cv.threshold(img2gray[3], 10, 255, cv.THRESH_BINARY)
#source = https://docs.opencv.org/3.4/d0/d86/tutorial_py_image_arithmetics.html

def has_alpha_opencv(img):
    #print (img.shape[-1])
    return img.shape[-1] == 4

def drawFps(t_start, frame,frame_index,last_reset_time):
    t_end = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (t_end - t_start)
    #print (F"T_end {t_end} T_start{t_start} and one minus the other{t_end-t_start}")
    fps_list.append(fps)
    if len(fps_list) > 10:
        del fps_list[0]
   
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (0, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 128, 26),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    frame_index += 1

    current_time = time.time()
    if current_time - last_reset_time >= 1.0:
        frame_index = 0
        last_reset_time = current_time
    return frame_index, last_reset_time

def shuffleBox(mainFrameWidth,mainFrameHeight,box_size):
    box_x = random.randint(0, mainFrameWidth - box_size[0])
    box_y = random.randint(0, mainFrameHeight - box_size[1])
    bbox = (box_x,box_y,box_size[0], box_size[1])
    return bbox

def isPinch (thumb_coordinates,index_coordinates,middle_coordinates,ring_coordinates,threshold=50):
     # Calculate distances between finger points
    thumb_index_distance = euclidean_distance(thumb_coordinates, index_coordinates)
    thumb_middle_distance = euclidean_distance(thumb_coordinates, middle_coordinates)
    thumb_ring_distance = euclidean_distance(thumb_coordinates, ring_coordinates)
    
    # Calculate the average distance between fingers
    average_distance = (thumb_index_distance + thumb_middle_distance + thumb_ring_distance) / 3
  
    # print (F"Index distance:{thumb_index_distance}/middle distance:{thumb_middle_distance}/ring distance:{thumb_ring_distance}")
    # print (F"Average Distance:{average_distance}")
    
    if average_distance<threshold : return True
    else : return False

def isPinchInsideBox( box,thumb_coordinates, index_coordinates, middle_coordinates,ring_coordinates, threshold=100):

    box_x, box_y, box_width, box_height = box

    #
    # Calculate distances between finger points
    thumb_index_distance = euclidean_distance(thumb_coordinates, index_coordinates)
    thumb_middle_distance = euclidean_distance(thumb_coordinates, middle_coordinates)
    thumb_ring_distance = euclidean_distance(thumb_coordinates, ring_coordinates)
   
    # Calculate the center of the box
    box_center_x = box_x + box_width / 2
    box_center_y = box_y + box_height / 2

    # print (F"thumbCoord:{thumb_coordinates}-indexCoord{index_coordinates} - box Coord{box}")
    # print (F"Box Center X {box_center_x} and Center Y{box_center_y}")
      
    # Needs to be in a try catch so it doesnt blow up if out of bounds
    try:
        # First calculate the distance between all the fingers and the thumb
        # then use the smallest one has reference, or the average
        
        # Calculate the distance between the index finger and the box center
        index_distance_to_center = ((index_coordinates[0] - box_center_x)**2 + (index_coordinates[1] - box_center_y)**2)**0.5
        middle_distance_to_center = ((middle_coordinates[0] - box_center_x)**2 + (middle_coordinates[1] - box_center_y)**2)**0.5
        ring_distance_to_center = ((ring_coordinates[0] - box_center_x)**2 + (ring_coordinates[1] - box_center_y)**2)**0.5

        miminum_distance_value = min(index_distance_to_center,middle_distance_to_center,ring_distance_to_center)
        # Check if the distance is within the threshold
        # print(F"Minimum Distance {miminum_distance_value}")
        return miminum_distance_value < threshold
    except Exception as e:
        # Catch any other exceptions
        print(f"An error occurred: {e}")
        return False
   
def displayScore(frame, score, difficulty):
    # Get the frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Define the font and position for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text = f"Difficulty: {difficulty}/Score: {score}"
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = (frame_width - text_size[0] - 10, frame_height - 10)

    # Put the text on the frame
    cv2.putText(frame, text, text_position, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return frame

def changeBoxSize(difficulty) :
    match difficulty:
        case 'Easy':
            box_size = (175,175)
        case 'Medium':
            box_size = (120,120)
        case 'Hard':
            box_size = (80,80)
        case 'God':
            box_size = (30,30)
        case _:
            box_size = (175,175)
    return box_size

# def get_right_hand_coordinates(hand_landmarks):
#     # Extract x and y coordinates of the right hand landmarks
#     right_hand_x = [hand_landmarks.landmark[i].x for i in range(9, 17)]
#     right_hand_y = [hand_landmarks.landmark[i].y for i in range(9, 17)]

#     return right_hand_x, right_hand_y   

def detect_thumb(frame,right_hand_landmarks):
    # Accessing the coordinates of the right thumb tip (landmark index 4)
    right_thumbtip = (right_hand_landmarks[4].x, right_hand_landmarks[4].y, right_hand_landmarks[4].z)

    # Accessing the coordinates of the right thumb base (landmark index 0)
    right_thumb_base = (right_hand_landmarks[0].x, right_hand_landmarks[0].y, right_hand_landmarks[0].z)

    # Draw a circle on the right thumb tip
    right_thumbtip_x = int(right_thumbtip[0] * frame.shape[1])
    right_thumbtip_y = int(right_thumbtip[1] * frame.shape[0])
    cv2.circle(frame, (right_thumbtip_x, right_thumbtip_y), 10, (0, 255, 0), -1)

    # Draw a circle on the right thumb base
    right_thumb_base_x = int(right_thumb_base[0] * frame.shape[1])
    right_thumb_base_y = int(right_thumb_base[1] * frame.shape[0])
    cv2.circle(frame, (right_thumb_base_x, right_thumb_base_y), 10, (0, 0, 255), -1)

    # Check for a more precise thumbs-up gesture
    thumb_length = right_hand_landmarks[4].x - right_hand_landmarks[0].x
    thumb_height = right_hand_landmarks[4].y - right_hand_landmarks[0].y

    # Define thresholds for thumbs-up and thumbs-down
    thumbs_up_threshold = 0.02
    thumbs_down_threshold = 0.02

    # Determine the gesture
    if thumb_length > thumbs_up_threshold and thumb_height < thumbs_down_threshold:
        return "UP"
    elif thumb_length > thumbs_down_threshold and thumb_height > thumbs_up_threshold:
        return "DOWN"
    else:
        return None

  
