import cv2
import os
import time
import functions as f
import random
import mediapipe as mp
from classes import DifficultyHandler


def main():
    # Set the path to the directory containing your image files
    directory_path = './images/'
    #initialize some variables
    last_reset_time=0
    frame_index=0
    score=0
    difficulty_handler = DifficultyHandler()
    difficulty = difficulty_handler.get_current_difficulty()
    pinchInside=False
    durationOfPinch=1
    
    # Define a threshold for pinch detection
    pinch_threshold = 0.02
    # Box size width and heigth
    box_size = f.changeBoxSize(difficulty)
    
    # Get a list of all files in the directory
    file_list = sorted(file for file in os.listdir(directory_path) if not file.startswith('.'))

    # Create an empty list to store frames
    framesWithBasketball = []

    # Read all images and store them in the frames list
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        frame = cv2.imread(file_path,cv2.IMREAD_UNCHANGED) # guarantees it has the original alpha channel
        framesWithBasketball.append(frame)

  # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Open the video capture
    cap = cv2.VideoCapture(1)  # Use 0 for the default camera, change if needed

    # Get the dimensions of the camera frame
    mainFrameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mainFrameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Generate random starting points for the bounding box
    bbox = f.shuffleBox(mainFrameWidth,mainFrameHeight,box_size)
    print (bbox)
    # Initialize time
    start_time = time.time()

    while True:
        #Deal with time
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= durationOfPinch:
            print("At least 1 second has passed.Releasing Pinch")
            pinchInside=False
            start_time = time.time()

        # Read a frame from the camera
        ret, frame = cap.read()
        t_start = cv2.getTickCount()

        if not ret:
            break  # Break the loop if reading the frame fails

        
        #MEDIAPIPE
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Holistic
        results = hands.process(rgb_frame)

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            # Extract landmarks for the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
            index_tip = hand_landmarks.landmark[8]  # Index finger tip

            # Calculate the distance between the thumb and index finger
            distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
           
            # Get the coordinates of thumb and index finger tips
            thumb_coordinates = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            index_coordinates = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
           
            # Check if the distance is below the threshold to detect a pinch
            if distance < pinch_threshold:
                #Draw "Pinch Detected" in the far right corner
                if (f.isPinchInsideBox(thumb_coordinates,index_coordinates,bbox)):
                # Draw "Pinch Detected" in the far right corner
                    print ("PINCH INSIDE BOX")
                    start_time = time.time()
                    pinchInside=True
                    #start the counter in time
                    cv2.putText(frame, "Pinch Detected Inside Box", (frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Pinch Detected", (frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #Just move the ball if there is an index
            if index_coordinates and pinchInside:
                bbox = *index_coordinates, *box_size

        #BASKETBALL
        # Display the corresponding image from the list inside the box
        fpsCalculations = f.drawFps(t_start,frame,frame_index,last_reset_time) # must return the frame index
        frame_index=fpsCalculations[0]
        last_reset_time=fpsCalculations[1]
        if frame_index < len(framesWithBasketball):
            img_to_display = framesWithBasketball[frame_index]

            # Resize the image to fit inside the box
            img_to_display = cv2.resize(img_to_display, (box_size[0], box_size[1]))

            #TODO: should test if the coordinates are valid. If they are not, should keep the position in bounds
            # Insert the image into the camera frame
            success = f.doMask(frame,bbox,img_to_display)
            if success is False :
                #reshuffle the basketball
                bbox = f.shuffleBox(mainFrameWidth,mainFrameHeight,box_size)
        #END FRAME PROCESSING
        #Display current Score and Difficulty
        f.displayScore(frame,score,difficulty)
        # Display the updated frame with the image
        cv2.imshow('Frame with Image', frame)
        # Exit the loop when the 'q' key is pressed or when all images are displayed
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
        if cv2.waitKey(1) & 0xFF == ord('w') :
            #shuffle the box position
            bbox = f.shuffleBox(mainFrameWidth,mainFrameHeight,box_size)
            print()
        if cv2.waitKey(1) & 0xFF == ord('d') :
            #change the difficulty and shuffle the box position
            difficulty = difficulty_handler.loop_difficulty()
            box_size = f.changeBoxSize(difficulty)
            bbox = f.shuffleBox(mainFrameWidth,mainFrameHeight,box_size)
            print()
            
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
