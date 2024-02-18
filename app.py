import cv2
import numpy as np

# Initialize Kalman filter parameters
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03

# Initialize video capture (Replace 'your_video.mp4' with the path to the provided video)
cap = cv2.VideoCapture('video.mp4')

# Initialize variables for dribble counting
prev_position = np.array([[0], [0]], dtype=np.float32)
dribble_count = 0
is_ball_upward = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for yellow color in HSV
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    # Threshold the image to get a binary mask of the yellow region
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply morphological operations to enhance ball detection
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)

            # Predict the next state using the Kalman filter
            prediction = kalman.predict()

            # Correct the state based on the measured values
            measurement = np.array([[x + w / 2], [y + h / 2]], dtype=np.float32)
            kalman.correct(measurement)

            # Count dribble if the ball moves upward
            if prev_position is not None and measurement[1] < prev_position[1] and not is_ball_upward:
                dribble_count += 1
                is_ball_upward = True
            elif prev_position is not None and measurement[1] > prev_position[1]:
                is_ball_upward = False

            prev_position = measurement

    # Display dribble count on the video
    cv2.putText(frame, f'Dribbles: {dribble_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Dribble Counting', frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Print the total number of dribbles
print(f'Total Dribbles: {dribble_count}')
