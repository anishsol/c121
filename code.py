import cv2
import mediapipe as mp

# Set up Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Set up webcam
cap = cv2.VideoCapture(0)

# Variables to track thumb states
thumbs_up = False
thumbs_down = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get hand landmarks
    results = hands.process(rgb_frame)

    # Check if hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmarks for thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Check the thumb and index finger Y-coordinates to determine gesture
            if thumb_tip.y < index_tip.y:
                thumbs_up = True
                thumbs_down = False
            else:
                thumbs_down = True
                thumbs_up = False

    # Display the result based on gesture
    if thumbs_up:
        cv2.putText(frame, "Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    elif thumbs_down:
        cv2.putText(frame, "Thumbs Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Gesture Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
