#import the libraries
import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,       
    max_num_hands=2,               
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5    
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )


    cv2.imshow('Camera', frame)


    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
