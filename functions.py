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
    else:
        print("Invalid box size")


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
  
