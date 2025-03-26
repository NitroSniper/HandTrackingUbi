import cv2
import mediapipe as mp
import csv
import time
import os

# configurations
SIGN_LABEL = "F"  # change label before running
NUM_SAMPLES = 100

# set up for csv file
filename = f"data/data_{SIGN_LABEL}.csv"
file_exists = os.path.isfile(filename)

# Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Webcam
cap = cv2.VideoCapture(0)
csv_file = open(filename, "a", newline='')
csv_writer = csv.writer(csv_file)

print(f"Collecting data for '{SIGN_LABEL}' (both hands)")
print("Starting in 3 seconds...")
time.sleep(3)

collected = 0
while collected < NUM_SAMPLES:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if (results.multi_hand_landmarks and
            results.multi_handedness and
            len(results.multi_hand_landmarks) == 2):

        # Initialise hand data dictionary
        hand_data = {'Left': [], 'Right': []}

        # Match landmarks to left/right
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            hand_data[hand_label] = landmarks

            # Draw on screen
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # if both hands detected, saved
        if hand_data['Left'] and hand_data['Right']:
            row = [SIGN_LABEL] + hand_data['Left'] + hand_data['Right']
            csv_writer.writerow(row)
            collected += 1
            print(f"Sample {collected}/{NUM_SAMPLES} collected")
            time.sleep(0.1)

    # display info
    cv2.putText(frame, f"Label: {SIGN_LABEL} | Collected: {collected}/{NUM_SAMPLES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Both Hands Sign Data", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
