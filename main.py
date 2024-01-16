import cv2
import os
import time
import functions as f
import random
import mediapipe as mp
from classes import DifficultyHandler


def main():
    # Set the path to the directory containing image files
    directory_path = './images/'
    congratulations_image_path = './images/congratulations.png'  # Replace with the actual path to your congratulations image
    basket_image_path = './images/basket.png'
    #initialize some variables
    last_reset_time=0
    frame_index=0
    #Scoring and congratulations variables
    score=0
    show_congratulations=False
    congratulations_display_duration = 40  # Number of loops to display the congratulations image
    congratulations_frame = cv2.imread(congratulations_image_path)
    loop_congratulations_count = 0
    
    basket_image = cv2.imread(basket_image_path)
    difficulty_handler = DifficultyHandler()
    difficulty = difficulty_handler.get_current_difficulty()
    pinchInside=False
    durationOfPinch=1
    #swipe variables
    left_swipe_counter = 0
    right_swipe_counter = 0
    swipe_threshold = 10 # this is actually the number of processed frames. Might need to be bigger
    last_index_x = None
    initial_index_x = None  # Record initial x-coordinate
    # Define a threshold for pinch detection
    pinch_threshold = 0.02
    # Box size width and heigth
    bb_box_size = f.changeBoxSize(difficulty)
    
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
    bb_box = f.shuffleBox(mainFrameWidth,mainFrameHeight,bb_box_size)
    print (bb_box)
    # Initialize time
    start_time = time.time()

    while True:
        #Deal with time
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= durationOfPinch:
            print("At least 1 second has passed.Releasing Pinch and reseting Swipes")
            pinchInside=False
            start_time = time.time()
            initial_index_x = None

        # Read a frame from the camera
        ret, frame = cap.read()
        t_start = cv2.getTickCount()

        if not ret:
            break  # Break the loop if reading the frame fails

        
        #MEDIAPIPE
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands
        results = hands.process(rgb_frame)

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            # Extract landmarks for the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
            index_tip = hand_landmarks.landmark[8]  # Index finger tip
            middle_tip = hand_landmarks.landmark[12] # middle finger tip
            ring_tip = hand_landmarks.landmark[16] # ring finger tip
             # Get the coordinates of thumb and index finger tips
            thumb_coordinates = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            index_coordinates = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
            middle_coordinates = (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0]))
            ring_coordinates = (int(ring_tip.x * frame.shape[1]), int(ring_tip.y * frame.shape[0]))

            # Test for pinch
            isPinch= f.isPinch(thumb_coordinates,index_coordinates,middle_coordinates,ring_coordinates)

            # Draw circles on the frame at each coordinate so we can see
            cv2.circle(frame, thumb_coordinates, 10, (0, 255, 0), -1)  # Green circle for thumb
            cv2.circle(frame, index_coordinates, 10, (0, 0, 255), -1)  # Red circle for index finger
            cv2.circle(frame, middle_coordinates, 10, (255, 0, 0), -1)  # Blue circle for middle finger
            cv2.circle(frame, ring_coordinates, 10, (255, 255, 0), -1)  # Yellow circle for ring finger

           
            # Check if the distance is below the threshold to detect a pinch
            if isPinch:
                #Draw "Pinch Detected" in the far right corner
                if (f.isPinchInsideBox(bb_box,thumb_coordinates,index_coordinates,middle_coordinates,ring_coordinates)):
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
                bb_box = *index_coordinates, *bb_box_size
                
            #CHANGE LEVEL USING SWIPES
            # Check if this is not the first frame
            if last_index_x is not None and pinchInside==False:
                # Record initial x-coordinate when the swipe starts
                if initial_index_x is None:
                    initial_index_x = index_tip.x
                    
                # Compare the current and last known index finger positions
                if index_tip.x < last_index_x:
                    left_swipe_counter += 1
                    right_swipe_counter = 0
                elif index_tip.x > last_index_x:
                    right_swipe_counter += 1
                    left_swipe_counter = 0
                    
                #calculate the pixels travelled
                percentage_travelled = abs(initial_index_x - index_tip.x)

                # Check for a left swipe
                if left_swipe_counter >= swipe_threshold and percentage_travelled>0.35:
                    print(F"Swipe Left detected! Looping Difficulty. % travelled{percentage_travelled}")
                    difficulty = difficulty_handler.loop_difficulty()
                    bb_box_size = f.changeBoxSize(difficulty)
                    bb_box = f.shuffleBox(mainFrameWidth,mainFrameHeight,bb_box_size)
                    left_swipe_counter = 0  # Reset the counter after detecting a left swipe
                    initial_index_x = None
                # Check for a right swipe
                if right_swipe_counter >= swipe_threshold and percentage_travelled>0.35:
                    print(F"Swipe Right detected! Looping Difficulty. % travelled{percentage_travelled}")
                    difficulty = difficulty_handler.loop_difficulty()
                    bb_box_size = f.changeBoxSize(difficulty)
                    bb_box = f.shuffleBox(mainFrameWidth,mainFrameHeight,bb_box_size)
                    right_swipe_counter = 0  # Reset the counter after detecting a right swipe
                    initial_index_x = None

            # Update the last known index finger position
            last_index_x = index_tip.x


        #BASKETBALL
        # Display the corresponding image from the list inside the box
        fpsCalculations = f.drawFps(t_start,frame,frame_index,last_reset_time) # must return the frame index
        frame_index=fpsCalculations[0]
        last_reset_time=fpsCalculations[1]
        if frame_index < len(framesWithBasketball):
            img_to_display = framesWithBasketball[frame_index]

            # Resize the image to fit inside the box
            img_to_display = cv2.resize(img_to_display, (bb_box_size[0], bb_box_size[1]))

            #TODO: should test if the coordinates are valid. If they are not, should keep the position in bounds
            # Insert the basketball image into the camera frame
            success = f.doMask(frame,bb_box,img_to_display)
            if success is False : # if the doMask reports invalid frame
                #reshuffle the basketball
                bb_box = f.shuffleBox(mainFrameWidth,mainFrameHeight,bb_box_size)
                
        #BASKET
        
        
        #END FRAME PROCESSING
    
        #Display current Score and Difficulty
        f.displayScore(frame,score,difficulty)
        # Display the updated frame with the image or the congratulations in case of scoring
        if (show_congratulations):
            cv2.imshow('Main Game', congratulations_frame)
            loop_congratulations_count += 1
            # Reset loop count after displaying the congratulations image for the specified duration
            if loop_congratulations_count >= congratulations_display_duration:
                loop_congratulations_count = 0
                show_congratulations=False
        else:
            cv2.imshow('Main Game', frame)
        # Exit the loop when the 'q' key is pressed or when all images are displayed
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
        if cv2.waitKey(1) & 0xFF == ord('s') :
            #shuffle the box position
            score+=1
            show_congratulations=True
        if cv2.waitKey(1) & 0xFF == ord('w') :
            #just shuffle the box position
            bb_box = f.shuffleBox(mainFrameWidth,mainFrameHeight,bb_box_size)
        if cv2.waitKey(1) & 0xFF == ord('d') :
            #change the difficulty and shuffle the box position
            difficulty = difficulty_handler.loop_difficulty()
            bb_box_size = f.changeBoxSize(difficulty)
            bb_box = f.shuffleBox(mainFrameWidth,mainFrameHeight,bb_box_size)
            
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
