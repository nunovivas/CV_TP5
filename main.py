import cv2
import os
import time
import functions as f

def main():
    # Set the path to the directory containing your image files
    directory_path = './images/'

    # Get a list of all files in the directory
    file_list = sorted(file for file in os.listdir(directory_path) if not file.startswith('.'))

    # Create an empty list to store frames
    framesWithBasketball = []
    fps=0

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

    # Define the size and position of the box in the middle of the camera frame
    box_size = (50, 50)
    box_x = (mainFrameWidth - box_size[0]) // 2
    box_y = (mainFrameHeight - box_size[1]) // 2
    bbox = (box_x,box_y,box_size[0], box_size[1])
    # Initialize variables for frame rate calculation
    start_time = time.time()
    frame_count = 0

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if reading the frame fails

    # Get the elapsed time in seconds
        elapsed_time = time.time() - start_time

        # Display the corresponding image from the list inside the box
        frame_index = int(elapsed_time * 30)  # Assuming 30 frames per second
        if frame_index < len(framesWithBasketball):
            img_to_display = framesWithBasketball[frame_index]

            # Resize the image to fit inside the box
            img_to_display = cv2.resize(img_to_display, (box_size[0], box_size[1]))

            # Insert the image into the camera frame
            #frame[box_y:box_y + box_size[1], box_x:box_x + box_size[0]] = img_to_display
            f.doMask(frame,bbox,img_to_display)
    
        # Display the dynamically calculated frame rate in the top-left corner
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:  # Update frame rate every second
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Exit the loop when the 'q' key is pressed or when all images are displayed
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
        # Display the updated frame with the image
        cv2.imshow('Frame with Image', frame)


    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
