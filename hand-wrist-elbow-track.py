import cv2
import mediapipe as mp
import math
import ikpy.chain
import ikpy.utils.plot as plot_utils
import numpy as np
import os
import matplotlib.pyplot as plt

# -------------------------
# MediaPipe
# -------------------------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# -------------------------
# Camera
# -------------------------
cap = cv2.VideoCapture(0)
cv2.namedWindow("Arm & Claw Gripper Tracking")

# -------------------------
# Load URDF
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, "actual_arm_urdf.urdf")

print("\nURDF path:", urdf_path)
print("URDF exists:", os.path.exists(urdf_path), "\n")

my_chain = ikpy.chain.Chain.from_urdf_file(
    urdf_path,
    active_links_mask=[False, True, True, True, True, True, False]
 
    # last three True = wrist_joint_2 + finger_left + finger_right
)

my_chain._base_pose = np.eye(4)

# -------------------------
# 3D Robot Viewer
# -------------------------
fig, ax = plot_utils.init_3d_figure()
fig.set_figwidth(8)
fig.set_figheight(6)
plt.ion()

def center_axes():
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 0.6)
    try:
        ax.set_box_aspect([1,1,1])
    except:
        pass

center_axes()

# -------------------------
# Robot target
# -------------------------
target_position = np.array([0.0, 0.2, 0.1])
filtered_target = target_position.copy()

TARGET_ALPHA = 0.15  # 0.1–0.2 is a good range

target_orientation = [-1, 0, 0]
ik = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="Y")

# -------------------------
# Gripper + wrist state
# -------------------------
grip_value = 0.0
wrist_twist = 0.0

# -------------------------
# Helpers
# -------------------------
def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def clamp(x, a, b):
    return max(a, min(b, x))

def remap(x, in_min, in_max, out_min, out_max):
    if abs(in_max - in_min) < 1e-6:
        return out_min
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# -------------------------
# IK + Plot update
# -------------------------
def doIK():
    global ik, grip_value, wrist_twist, frame_count

    ik = my_chain.inverse_kinematics(
        target_position,
        target_orientation,
        orientation_mode="Y",
        initial_position=ik
    )

    ik[-3] = clamp(wrist_twist, -3.14, 3.14)
    ik[-2] = clamp(grip_value, 0.0, 1.0)

    frame_count += 1
    if frame_count % PLOT_EVERY_N == 0:
        ax.cla()
        center_axes()
        my_chain.plot(ik, ax, target=target_position)
        plt.pause(0.001)


def move(x, y, z):
    global target_position, filtered_target

    new_target = np.array([x, y, z])

    filtered_target = (
        TARGET_ALPHA * new_target
        + (1.0 - TARGET_ALPHA) * filtered_target
    )

    target_position = filtered_target.tolist()
    doIK()


# -------------------------
# Models
# -------------------------
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)


frame_count = 0
PLOT_EVERY_N = 2  # draw every 2 frames (~30 FPS)

smoothed_neck = None
NECK_ALPHA = 0.2



# -------------------------
# Main loop
# -------------------------
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb)
    hand_results = hands.process(rgb)

    # -------------------------
    # Neck reference
    # -------------------------
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        new_neck = np.array([
            (ls.x + rs.x) / 2,
            (ls.y + rs.y) / 2,
            (ls.z + rs.z) / 2
        ])

        if smoothed_neck is None:
            smoothed_neck = new_neck
        else:
            smoothed_neck = (
                NECK_ALPHA * new_neck
                + (1.0 - NECK_ALPHA) * smoothed_neck
            )

        neck_ref = {
            "x": smoothed_neck[0],
            "y": smoothed_neck[1],
            "z": smoothed_neck[2]
        }

        px = (int(neck_ref["x"] * w), int(neck_ref["y"] * h))
        cv2.circle(image, px, 10, (255, 255, 0), -1)


    # -------------------------
    # Hand tracking
    # -------------------------
    if hand_results.multi_hand_landmarks and neck_ref:
        hand = hand_results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

        thumb = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring = hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky = hand.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # Gripper
        pinch = calculate_distance(
            (thumb.x, thumb.y),
            (index.x, index.y)
        )
        grip_value = remap(pinch, 0.02, 0.12, 0.0, 1.0)
        grip_value = clamp(grip_value, 0.0, 1.0)


        # Wrist rotation
        ix, iy = index.x, index.y
        px, py = pinky.x, pinky.y
        hand_angle = math.atan2(py - iy, px - ix)
        wrist_twist = clamp(hand_angle, -1.5, 1.5)

        # End-effector XYZ
        fingers_x = (index.x + middle.x + ring.x + pinky.x) / 4
        fingers_y = (index.y + middle.y + ring.y + pinky.y) / 4
        fingers_z = (index.z + middle.z + ring.z + pinky.z) / 4

        fingers_rel = {
            "x": fingers_x - neck_ref["x"],
            "y": neck_ref["y"] - fingers_y,
            "z": fingers_z - neck_ref["z"]
        }
        z_scaled = -fingers_rel["z"] * 0.5 + 0.3

        move(fingers_rel["x"], z_scaled, fingers_rel["y"])

    cv2.imshow("Arm & Claw Gripper Tracking", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
pose.close()
hands.close()

cap.release()
cv2.destroyAllWindows()
