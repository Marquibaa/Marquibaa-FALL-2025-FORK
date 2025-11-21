import cv2
import mediapipe as mp
import math
from collections import deque
import numpy as np

# -------------------------
# ★ NEW: IKPY IMPORTS
# -------------------------
# Make sure ikpy is installed: pip install ikpy
import ikpy.chain
# -------------------------

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# -------------------------
# ★ ADD: SMOOTHING BUFFERS
# -------------------------
SMOOTHING_WINDOW = 5
smooth_buffers = {
    "thumb": deque(maxlen=SMOOTHING_WINDOW),
    "finger_center": deque(maxlen=SMOOTHING_WINDOW)
}

def smooth_point(buffer, new_point):
    """Average last N points for smoother tracking."""
    buffer.append(np.array(new_point))
    return tuple(np.mean(buffer, axis=0).astype(int))
# -----------------------------------------------------

# -------------------------
# ★ NEW: IK CHAIN SETUP
# -------------------------
# Update this path to your URDF file location
URDF_PATH = "arm_urdf.urdf"

# active_links_mask: set according to your URDF (first link inactive if base is fixed)
try:
    my_chain = ikpy.chain.Chain.from_urdf_file(URDF_PATH, active_links_mask=[False, True, True, True, True, True])
except Exception as e:
    print(f"Warning: unable to load URDF from '{URDF_PATH}'. IK disabled. Error: {e}")
    my_chain = None

# initial IK solution (will be set once IK runs)
ik_solution = None

# Mapping / calibration parameters (tweak these to match your real robot)
SCALE_X = 0.5   # meters per normalized unit in x
SCALE_Y = 0.5   # meters per normalized unit in y
SCALE_Z = 0.5   # meters per normalized unit in z

# Offsets to translate camera-origin (neck) to robot base frame if needed
OFFSET_X = 0.0
OFFSET_Y = 0.0
OFFSET_Z = 0.0

# Default target (meters)
target_position = [0.0, 0.0, 0.2]
target_orientation = [-1, 0, 0]  # keep as in your notebook example
# -----------------------------------------------------

# Initialize webcam
cap = cv2.VideoCapture(0)

# Fullscreen setup
cv2.namedWindow('Arm & Claw Gripper Tracking', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Arm & Claw Gripper Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Arm landmarks
arm_landmarks = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_WRIST,
]

# Arm connections
arm_connections = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
]

def calculate_distance(point1, point2):
    """Calculate distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# small helper to call IK safely
def compute_ik(target_pos, initial_position=None):
    global my_chain
    if my_chain is None:
        return None
    try:
        # orientation_mode "Y" used in your notebook example
        angles = my_chain.inverse_kinematics(target_pos, target_orientation, orientation_mode="Y", initial_position=initial_position)
        return angles
    except Exception as e:
        print("IK error:", e)
        return None

# Configure MediaPipe
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip for selfie-view
        image = cv2.flip(image, 1)
        h, w, c = image.shape
        
        # Single RGB conversion
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        pose_results = pose.process(image_rgb)
        hand_results = hands.process(image_rgb)
        
        # Draw ARM tracking
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Draw arm connections
            for connection in arm_connections:
                start = landmarks[connection[0]]
                end = landmarks[connection[1]]
                start_px = (int(start.x * w), int(start.y * h))
                end_px = (int(end.x * w), int(end.y * h))
                cv2.line(image, start_px, end_px, (0, 255, 0), 3)
            
            # Draw arm landmarks
            for landmark_idx in arm_landmarks:
                landmark = landmarks[landmark_idx]
                px = (int(landmark.x * w), int(landmark.y * h))
                cv2.circle(image, px, 8, (0, 0, 255), -1)
                cv2.circle(image, px, 10, (255, 255, 255), 2)
        
        # Get neck reference point (average of shoulders)
        neck_ref = None
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Calculate neck midpoint
            neck_ref = {
                'x': (left_shoulder.x + right_shoulder.x) / 2,
                'y': (left_shoulder.y + right_shoulder.y) / 2,
                'z': (left_shoulder.z + right_shoulder.z) / 2
            }
            
            # Draw neck reference
            neck_px = (int(neck_ref['x'] * w), int(neck_ref['y'] * h))
            cv2.circle(image, neck_px, 10, (255, 255, 0), -1)
            cv2.circle(image, neck_px, 12, (255, 255, 255), 2)
            cv2.putText(image, "NECK (0,0,0)", (neck_px[0] + 15, neck_px[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw HAND and Claw
        if hand_results.multi_hand_landmarks and neck_ref:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                
                # Draw subtle hand skeleton
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(120, 120, 120), thickness=1))
                
                # Handedness
                handedness = hand_results.multi_handedness[hand_idx].classification[0].label

                # Extract raw fingertip positions
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Convert to pixel coords
                thumb_px_raw = (int(thumb.x * w), int(thumb.y * h))
                fingers_center_raw = (
                    int(((index.x + middle.x + ring.x + pinky.x) / 4) * w),
                    int(((index.y + middle.y + ring.y + pinky.y) / 4) * h)
                )

                # --------------------------------------------
                # ★ APPLY SMOOTHING HERE
                # --------------------------------------------
                thumb_px = smooth_point(smooth_buffers["thumb"], thumb_px_raw)
                fingers_center = smooth_point(smooth_buffers["finger_center"], fingers_center_raw)
                # --------------------------------------------

                # Relative coordinates (unchanged)
                fingers_center_norm = {
                    "x": (index.x + middle.x + ring.x + pinky.x) / 4,
                    "y": (index.y + middle.y + ring.y + pinky.y) / 4,
                    "z": (index.z + middle.z + ring.z + pinky.z) / 4
                }

                thumb_rel = {
                    "x": thumb.x - neck_ref['x'],
                    "y": thumb.y - neck_ref['y'],
                    "z": thumb.z - neck_ref['z']
                }

                fingers_rel = {
                    "x": fingers_center_norm['x'] - neck_ref['x'],
                    "y": fingers_center_norm['y'] - neck_ref['y'],
                    "z": fingers_center_norm['z'] - neck_ref['z']
                }

                # Claw distance
                claw_distance = calculate_distance(thumb_px, fingers_center)

                # Draw jaws
                cv2.circle(image, thumb_px, 15, (0, 0, 255), -1)
                cv2.circle(image, thumb_px, 17, (255, 255, 255), 2)

                cv2.circle(image, fingers_center, 15, (255, 0, 0), -1)
                cv2.circle(image, fingers_center, 17, (255, 255, 255), 2)

                # Line between jaws
                cv2.line(image, thumb_px, fingers_center, (0, 255, 255), 3)

                # Distance label
                midx = (thumb_px[0] + fingers_center[0]) // 2
                midy = (thumb_px[1] + fingers_center[1]) // 2
                cv2.putText(image, f"{claw_distance:.0f}px", (midx + 10, midy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # -------------------------
                # ★ NEW: MAP TO ROBOT COORDINATES
                # -------------------------
                # MediaPipe uses normalized coords relative to image: x right, y down, z negative forward (camera dependent)
                # We map neck-centered normalized coordinates -> meters for IK target.
                # You should tune SCALE_* and OFFSET_* to match your robot workspace and camera calibration.
                mapped_x = fingers_rel['x'] * SCALE_X + OFFSET_X
                mapped_y = -fingers_rel['y'] * SCALE_Y + OFFSET_Y   # invert y so up in camera -> positive y in robot (common convention)
                mapped_z = -fingers_rel['z'] * SCALE_Z + OFFSET_Z   # invert z so hand forward -> positive z (tweak as required)

                # Update target position (3-element list)
                target_position = [mapped_x, mapped_y, mapped_z]

                # Optionally clamp Z to safe limits so robot doesn't try unreachable negative heights
                if target_position[2] < 0.0:
                    target_position[2] = 0.0

                # Compute IK (use previous ik_solution as initial_position for continuity)
                new_ik = None
                if my_chain is not None:
                    try:
                        # Ensure target_position is 3 floats. ikpy expects a 3-vector for position.
                        # When calling inverse_kinematics we pass orientation and optionally initial_position
                        if ik_solution is None:
                            new_ik = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="Y")
                        else:
                            new_ik = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="Y", initial_position=ik_solution)
                        ik_solution = new_ik
                    except Exception as e:
                        print("IK compute failed:", e)
                        new_ik = None

                # Display joint angles (degrees) on the side panel if available
                joint_text = "IK: n/a"
                if ik_solution is not None:
                    # ik_solution is an array of angles (radians) including inactive joint at index 0
                    joint_angles_deg = list(map(lambda r: math.degrees(r), ik_solution.tolist()))
                    joint_text = "IK (deg): " + ", ".join([f"{ang:.1f}" for ang in joint_angles_deg])
                    
                    # show first 6 active joints on panel (or fewer if your chain smaller)
                    display_angles = joint_angles_deg[:7]  # adjust slice to match chain length
                else:
                    display_angles = None

                # SIDE PANEL INFO (unchanged layout, now includes IK)
                panel_x = 10 if handedness == "Left" else w - 400
                panel_y = h - 220
                
                cv2.rectangle(image, (panel_x - 5, panel_y - 25), 
                             (panel_x + 395, panel_y + 185), (0, 0, 0), -1)
                cv2.rectangle(image, (panel_x - 5, panel_y - 25), 
                             (panel_x + 395, panel_y + 185), (255, 255, 255), 2)
                
                cv2.putText(image, f"{handedness} Hand - Relative to Neck", 
                           (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(image, f"Jaw 1 (Thumb):", 
                           (panel_x, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(image, f"  X: {thumb_rel['x']:+.3f}  Y: {thumb_rel['y']:+.3f}  Z: {thumb_rel['z']:+.3f}", 
                           (panel_x, panel_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                
                cv2.putText(image, f"Jaw 2 (Fingers):", 
                           (panel_x, panel_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(image, f"  X: {fingers_rel['x']:+.3f}  Y: {fingers_rel['y']:+.3f}  Z: {fingers_rel['z']:+.3f}", 
                           (panel_x, panel_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                
                cv2.putText(image, f"Distance: {claw_distance:.0f}px", 
                           (panel_x, panel_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # IK display (concise)
                cv2.putText(image, joint_text, (panel_x, panel_y + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

                # Print to console (useful for debugging)
                if ik_solution is not None:
                    print("Target (m):", [f"{v:.3f}" for v in target_position], " | IK (deg):", ", ".join([f"{a:.1f}" for a in joint_angles_deg]))
                    # Uncomment and adapt the next line to send servo commands to hardware.
                    # sendCommand(ik_solution[1].item(), ik_solution[2].item(), ik_solution[3].item(), ik_solution[4].item(), ik_solution[5].item(), ik_solution[6].item(), 1)

        # Display
        cv2.imshow('Arm & Claw Gripper Tracking', image)
        
        # Exit on 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
