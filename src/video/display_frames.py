import cv2

image_path = "outputs/frames/frame_0000.jpg"

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

# Display image
cv2.imshow("Football Video Frame", image)

# Wait until a key is pressed
cv2.waitKey(0)

# Close window
cv2.destroyAllWindows()