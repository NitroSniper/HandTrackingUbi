import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utilities import draw_landmarks_on_image


def main():
    capture = cv2.VideoCapture(0)

    base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)


    while cv2.waitKey(1) != ord('q'):
        _, frame = capture.read()

        x, y, z = frame.shape

        frame = cv2.flip(frame, 1)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(image)

        # STEP 5: Process the classification result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        cv2.imshow("Picture", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


        cv2.imshow("Output", frame)



if __name__ == '__main__':
    main()