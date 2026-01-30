# coding: utf-8

import ikpy.chain
import numpy as np
import math
import serial
import time
import serial.tools.list_ports
import cv2
import mediapipe as mp


# -------------------------
# MediaPipe Setup
# -------------------------

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)


# -------------------------
# Depth Calibration
# -------------------------

DEPTH_MIN = None
DEPTH_MAX = None

start_time = time.time()

CALIB_FAR_TIME = 8
CALIB_CLOSE_TIME = 8


# -------------------------
# IK Setup
# -------------------------

my_chain = ikpy.chain.Chain.from_urdf_file(
    "Hardware/actual_arm_urdf.urdf",
    active_links_mask=[False, True, True, True, True, True]
)

target_position = [0, 0.2, 0.1]
target_orientation = [-1, 0, 0]

ik = my_chain.inverse_kinematics(
    target_position,
    target_orientation,
    orientation_mode="Y"
)

    
claw_state = "HAND NOT DETECTED"

# -------------------------
# Arduino Setup
# -------------------------

def find_arduino():

    ports = list(serial.tools.list_ports.comports())

    for p in ports:
        if ("Arduino" in p.description) or ("CH340" in p.description):
            return p.device

    return None


arduino_port = find_arduino()

if not arduino_port:
    print("No Arduino found")
    exit()

ser = serial.Serial(arduino_port, 9600, timeout=1)
time.sleep(2)


# -------------------------
# Motor Control
# -------------------------

def rad_to_motor_degree(rad, rad_min, rad_max, motor_min=0, motor_max=180):

    rad = max(min(rad, rad_max), rad_min)

    normalized = (rad - rad_min) / (rad_max - rad_min)

    return int(motor_min + normalized * (motor_max - motor_min))


def _send_packet(servo_vals, stepper_val):

    data = " ".join(map(str, servo_vals)) + " " + str(stepper_val) + "\n"
    ser.write(data.encode())


last_sent = 0
last_base_value = 0.0


def send_all_motors(angle1, angle2, angle3, angle4, angle5, value):

    global last_sent, last_base_value

    now = time.time()

    if now - last_sent < 0.1:
        return

    base_current = int(rad_to_motor_degree(last_base_value, 0, 4.71, 0, 270))
    base_target = int(rad_to_motor_degree(value, 0, 4.71, 0, 270))

    neutral_servos = [
        int(math.degrees(0.2)),
        rad_to_motor_degree(0.0, -1.92, 1.22),
        rad_to_motor_degree(0.0, -1.92, 0.7),
        rad_to_motor_degree(0.0, -1.39, 1.57),
        rad_to_motor_degree(0.0, -1.39, 1.57)
    ]

    _send_packet(neutral_servos, base_current)
    time.sleep(2)

    if base_target != base_current:
        _send_packet(neutral_servos, base_target)
        time.sleep(3)

    target_servos = [
        int(math.degrees(angle1)),
        rad_to_motor_degree(angle2, -1.92, 1.22),
        rad_to_motor_degree(angle3, -1.92, 0.7),
        rad_to_motor_degree(angle4, -1.39, 1.57),
        rad_to_motor_degree(angle5, -1.39, 1.57)
    ]

    _send_packet(target_servos, base_target)

    last_base_value = value
    last_sent = now


# -------------------------
# IK Wrapper
# -------------------------

def doIK():

    global ik

    old_position = ik.copy()

    ik = my_chain.inverse_kinematics(
        target_position,
        target_orientation,
        orientation_mode=None,
        initial_position=old_position
    )


def move(x, y, z):

    global target_position

    target_position = [x, y, z]

    doIK()

    send_all_motors(
        0.2,
        ik[5].item(),
        ik[4].item(),
        ik[3].item(),
        ik[2].item(),
        ik[1].item()
    )


# -------------------------
# Camera Loop
# -------------------------

frame_count = 0
PLOT_EVERY_N = 120

print("Starting camera. Calibrate first.")


while cap.isOpened():

    success, image = cap.read()

    if not success:
        continue

    image = cv2.flip(image, 1)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb)
    hand_results = hands.process(rgb)

    elapsed = time.time() - start_time


    # -------------------------
    # Calibration Status
    # -------------------------

    if elapsed < CALIB_FAR_TIME:
        status = "MOVE HAND FAR"

    elif elapsed < CALIB_FAR_TIME + CALIB_CLOSE_TIME:
        status = "MOVE HAND CLOSE"

    else:
        status = "CALIBRATION DONE"


    neck_ref = None

    # -------------------------
    # Neck Reference
    # -------------------------

    if pose_results.pose_landmarks:

        lm = pose_results.pose_landmarks.landmark

        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        neck_ref = {
            "x": (ls.x + rs.x) / 2,
            "y": (ls.y + rs.y) / 2,
            "z": (ls.z + rs.z) / 2
        }


    # -------------------------
    # Hand Tracking
    # -------------------------

    if hand_results.multi_hand_landmarks and neck_ref:

        hand = hand_results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            image,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        lm = hand.landmark

        wrist = lm[0]
        index = lm[8]
        middle = lm[12]
        ring = lm[16]
        pinky = lm[20]


        # -------------------------
        # Your Axis Fix
        # -------------------------

        fingers_x = (index.x + middle.x + ring.x + pinky.x) / 4
        fingers_z = (index.y + middle.y + ring.y + pinky.y) / 4
        fingers_y = (index.z + middle.z + ring.z + pinky.z) / 4

        #FOR THE CLAW

        

        thumb = lm[4]
        thumb_x = thumb.x
        thumb_y = thumb.y

        hand_opening = math.dist(
            [index.x, index.y],
            [thumb_x, thumb_y]
        )

        if hand_opening > 0.05:
            claw_state = "CLAW OPEN"
        else:
            claw_state = "CLAW CLOSED"


        # -------------------------
        # Hand Size (Depth)
        # -------------------------

        hand_size = math.dist(
            [wrist.x, wrist.y],
            [index.x, index.y]
        )


        # FAR
        if elapsed < CALIB_FAR_TIME:

            if DEPTH_MIN is None or hand_size < DEPTH_MIN:
                DEPTH_MIN = hand_size


        # CLOSE
        elif elapsed < CALIB_FAR_TIME + CALIB_CLOSE_TIME:

            if DEPTH_MAX is None or hand_size > DEPTH_MAX:
                DEPTH_MAX = hand_size


        # -------------------------
        # Depth Mapping
        # -------------------------

        depth_val = 0.7

        if DEPTH_MIN and DEPTH_MAX:

            if DEPTH_MAX - DEPTH_MIN > 1e-4:

                depth_val = (hand_size - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
                depth_val = np.clip(depth_val, 0, 1)

                depth_val = 0.5 + depth_val * 0.5


        # -------------------------
        # Relative Position
        # -------------------------

        fingers_rel = {
            "x": fingers_x - neck_ref["x"],
            "y": depth_val,
            "z": fingers_z - neck_ref["z"]
        }


        x_scaled = -fingers_rel["x"]
        y_scaled = fingers_rel["y"] *- 1 + 0.5
        z_scaled = fingers_rel["z"] *-1 + 1


        # -------------------------
        # Move Robot
        # -------------------------

        frame_count += 1

        if frame_count % PLOT_EVERY_N == 0:

            move(x_scaled, y_scaled, z_scaled)

            print(
                f"Target X:{x_scaled:.2f} "
                f"Y:{y_scaled:.2f} "
                f"Z:{z_scaled:.2f}"
            )


        # Depth Display
        cv2.putText(
            image,
            f"Depth: {y_scaled:.2f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )


    # Status
    cv2.putText(
        image,
        f'{status} {claw_state}',
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,255),
        2
    )



    cv2.imshow("Robot Control", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -------------------------
# Cleanup
# -------------------------

cap.release()
cv2.destroyAllWindows()

pose.close()
hands.close()
