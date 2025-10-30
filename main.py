#import the libraries
import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(
    static_image_mode=False,       
    max_num_hands=2,               
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5    
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(rgb)
    result_pose = pose.process(rgb)

    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
    
    if result_pose.pose_landmarks:
        landmarks = result_pose.pose_landmarks.landmark
        
        h, w, _ = frame.shape
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

        x_shoulder, y_shoulder = int(right_shoulder.x * w), int(right_shoulder.y * h)
        x_elbow, y_elbow = int(right_elbow.x * w), int(right_elbow.y * h)

        cv2.circle(frame, (x_shoulder, y_shoulder), 8, (0, 255, 0), -1)
        cv2.circle(frame, (x_elbow, y_elbow), 8, (0, 255, 0), -1)

    cv2.imshow('Camera', frame)


    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
