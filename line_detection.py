import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    # Capture frame from the video
    _, img = cap.read()
    if img is None:
        print("Error: Image not found!")
        break

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([22, 12, 0])
    upper = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img, img, mask=mask)

    # Edge detection
    edges = cv2.Canny(res, 150, 250, apertureSize=3)

    # Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)  # Adjust threshold if needed

    if lines is not None:
        # Draw the lines
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(res, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Detected Lines', res)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
