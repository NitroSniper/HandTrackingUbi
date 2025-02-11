from time import perf_counter

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utilities import draw_landmarks_on_image


annotated_image = None

def print_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global annotated_image
    # print('hand landmarker result: {}'.format(result))
    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)

def main():
    capture = cv2.VideoCapture(0)

    base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2, running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=print_result
    )

    with vision.HandLandmarker.create_from_options(options) as detector:
        while cv2.waitKey(1) != ord('q'):
            _, frame = capture.read()

            x, y, z = frame.shape

            frame = cv2.flip(frame, 1)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detector.detect_async(image, int(perf_counter()*1000))

            if annotated_image is not None:
                cv2.imshow("Picture", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            # STEP 5: Process the classification result. In this case, visualize it.
            # cv2.imshow("Output", frame)



if __name__ == '__main__':
    main()