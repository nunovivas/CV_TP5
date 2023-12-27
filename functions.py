import cv2 as cv2

def doMask(frame, box, newImage):
    # Resize the image to fit the bounding box
    print (box)
    newImage_resized = cv2.resize(newImage, (box[2], box[3]))
    # Print shapes for troubleshooting
    print("Image resized shape:", newImage_resized.shape)
    print("Box:", box)
    print("Frame shape:", frame.shape[0], frame.shape[1])

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
    print (img.shape[-1])
    return img.shape[-1] == 4