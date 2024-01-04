import random
import cv2 as cv2
from statistics import mean as mean
import time


# Store fps values for calculating mean
fps_list = []

def doMask(frame, box, newImage):
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
                            (1.0 - alpha_channel) * roi[:, :, c])

        # Update the frame with the blended result
        frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = roi
        return True
    else:
        print("Invalid box size")
        return False


def has_alpha_opencv(img):
    #print (img.shape[-1])
    return img.shape[-1] == 4

def drawFps(t_start, frame,frame_index,last_reset_time):
    t_end = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (t_end - t_start) / 1000
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
def isPinchInsideBox(thumb_coordinates, index_coordinates, box, threshold=100):
    thumb_x, thumb_y = thumb_coordinates
    index_x, index_y = index_coordinates

    box_x, box_y, box_width, box_height = box

    # Calculate the center of the box
    box_center_x = box_x + box_width / 2
    box_center_y = box_y + box_height / 2

    print (F"thumbCoord:{thumb_coordinates}-indexCoord{index_coordinates} - box Coord{box}")
    print (F"Box Center X {box_center_x} and Center Y{box_center_y}")
      
    # Needs to be in a try catch so it doesnt blow up if out of bounds
    try:
        # Calculate the distance between the index finger and the box center
        distance_to_center = ((index_x - box_center_x)**2 + (index_y - box_center_y)**2)**0.5
        # Check if the distance is within the threshold
        print(F"Distance {distance_to_center}")
        return distance_to_center < threshold
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
            box_size = (150,150)
        case 'Hard':
            box_size = (125,125)
        case 'God':
            box_size = (100,100)
        case _:
            box_size = (175,175)
    return box_size

        
  
