import cv2
import os
import time
import functions as f
import random

def main():
    # Set the path to the directory containing your image files
    directory_path = './images/'
    last_reset_time=0
    frame_index=0
    
    # Get a list of all files in the directory
    file_list = sorted(file for file in os.listdir(directory_path) if not file.startswith('.'))

    # Create an empty list to store frames
    framesWithBasketball = []

    # Read all images and store them in the frames list
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        frame = cv2.imread(file_path,cv2.IMREAD_UNCHANGED) # guarantees it has the original alpha channel
        framesWithBasketball.append(frame)

    # Open the video capture
    cap = cv2.VideoCapture(1)  # Use 0 for the default camera, change if needed

    # Get the dimensions of the camera frame
    mainFrameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mainFrameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    box_size = (50,50)
    # Generate random starting points for the bounding box
    bbox = f.shuffleBox(mainFrameWidth,mainFrameHeight,box_size)
    print (bbox)
    # Initialize variables for frame rate calculation

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        t_start = cv2.getTickCount()

        if not ret:
            break  # Break the loop if reading the frame fails


        # Display the corresponding image from the list inside the box
        fpsCalculations = f.drawFps(t_start,frame,frame_index,last_reset_time) # must return the frame index
        frame_index=fpsCalculations[0]
        last_reset_time=fpsCalculations[1]
        if frame_index < len(framesWithBasketball):
            img_to_display = framesWithBasketball[frame_index]

            # Resize the image to fit inside the box
            img_to_display = cv2.resize(img_to_display, (box_size[0], box_size[1]))

            # Insert the image into the camera frame
            f.doMask(frame,bbox,img_to_display)
    
        # Display the updated frame with the image
        cv2.imshow('Frame with Image', frame)
        # Exit the loop when the 'q' key is pressed or when all images are displayed
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
        if cv2.waitKey(1) & 0xFF == ord('w') :
            #shuffle the box position
            bbox = f.shuffleBox(mainFrameWidth,mainFrameHeight,box_size)
            print()
            
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
