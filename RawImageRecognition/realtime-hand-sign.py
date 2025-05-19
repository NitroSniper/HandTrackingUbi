import cv2
import numpy as np
import mediapipe as mp
import pickle
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
#%%
# configuration
model_path = "neigh-model.pkl"  # path to the saved model
data_direct   = "images"
image_size   = (244, 244) # resize image
# dominant_hand = "Right" # can be changed (but need to train the new model)
#%%
# Load knn model and labelEncoder
with open(model_path, 'rb') as f:
    model, label = pickle.load(f)
#%%
def process_raw_features(raw_features: mp.tasks.vision.HandLandmarkerResult):
    landmarks = []
    for hand_landmarks in results.hand_landmarks:
        x_coordinates = np.array([landmark.x for landmark in hand_landmarks])
        x_min, x_max = x_coordinates.min(), x_coordinates.max()
        norm_x_coordinates = (x_coordinates - x_min) / (x_max - x_min)
        y_coordinates = np.array([landmark.y for landmark in hand_landmarks])
        y_min, y_max = y_coordinates.min(), y_coordinates.max()
        norm_y_coordinates = (y_coordinates - y_min) / (y_max - y_min)
        z_coordinates = np.array([landmark.z for landmark in hand_landmarks])

        landmarks.append(np.array([
            [x, y, z] for x, y, z in zip(norm_x_coordinates, norm_y_coordinates,     z_coordinates)
        ]).flatten())
    return landmarks
#%%


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result, label):
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for hand_landmarks, handedness in zip(detection_result.hand_landmarks, detection_result.handedness):

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        # solutions.drawing_utils.draw_landmarks(
        #     annotated_image,
        #     hand_landmarks_proto,
        #     solutions.hands.HAND_CONNECTIONS,
        #     solutions.drawing_styles.get_default_hand_landmarks_style(),
        #     solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{label}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

#%%
def process_and_evaluate(raw_features: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image):
    X = process_raw_features(raw_features)
    if len(X) == 0:
        return output_image.numpy_view()
    result = label.classes_[model.predict(X)]
    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), raw_features, result)
    return annotated_image
#%%
# load the trained model
base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options, num_hands=1, running_mode=mp.tasks.vision.RunningMode.IMAGE,
    # result_callback=process_and_evaluate
)

#%%
with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
    capture = cv2.VideoCapture(0)
    while cv2.waitKey(100) != ord('q'):
        _, frame = capture.read()

        frame = cv2.flip(frame, 1)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = detector.detect(image)
        output_image = process_and_evaluate(results, image)

        cv2.imshow("Picture", output_image)
