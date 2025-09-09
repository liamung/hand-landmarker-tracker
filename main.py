import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

latest_result = None

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

  return annotated_image

model_path = "models/hand_landmarker.task"

HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode

'''
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    frame = output_image.numpy_view()
    annotated_img = draw_landmarks_on_image(frame, result)
    cv.imshow("Hand Landmarks", cv.cvtColor(annotated_img, cv.COLOR_RGB2BGR))
    cv.waitKey(1)
'''
def result_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
   global latest_result
   latest_result = (result, output_image)

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    num_hands=2,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    timestamp_ms = int(time.time() * 1000)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    landmarker.detect_async(mp_image, timestamp_ms)
 
    # Our operations on the frame come here
    # Display the resulting frame
    #annotated_img = draw_landmarks_on_image(mp_image.numpy_view(), det_result)
    #frame = cv.flip(frame, 1)
    #cv.imshow(cv.cvtColor(annotated_img, cv.COLOR_RGB2BGR))

    annotated_img = cv.flip(frame, 1)

    if latest_result:
       result, output_image = latest_result
       annotated_img = draw_landmarks_on_image(output_image.numpy_view(), result)
       cv.imshow("Hand Landmarks", cv.cvtColor(annotated_img, cv.COLOR_RGB2BGR))
    else:
       cv.imshow("Hand Landmarks", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()