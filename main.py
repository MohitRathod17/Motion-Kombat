import cv2
import time
import numpy as np
from ultralytics import YOLO
import vgamepad as vg

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12),
    (11, 13), (12, 14), (13, 15), (14, 16)
]

model = YOLO("yolov8n-pose.pt")
gamepad = vg.VX360Gamepad()
cap = cv2.VideoCapture(0)

facing_mode = 'right'
last_toggle_time = 0
toggle_cooldown = 1.5
gesture_cooldown = 0.8
last_action_time = 0
hand_raise_state = {"left": False, "right": False}
combo2_triggered = False

scorpion_combo_1 = {
    'left': [vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT, vg.XUSB_BUTTON.XUSB_GAMEPAD_X, vg.XUSB_BUTTON.XUSB_GAMEPAD_Y],
    'right': [vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT, vg.XUSB_BUTTON.XUSB_GAMEPAD_X, vg.XUSB_BUTTON.XUSB_GAMEPAD_Y],
}

scorpion_combo_2 = {
    'left': [vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT, vg.XUSB_BUTTON.XUSB_GAMEPAD_X],
    'right': [vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT, vg.XUSB_BUTTON.XUSB_GAMEPAD_X],
}

def press_combo(buttons):
    for btn in buttons:
        gamepad.press_button(button=btn)
    gamepad.update()
    time.sleep(0.2)
    for btn in buttons:
        gamepad.release_button(button=btn)
    gamepad.update()

def press_and_release(button=None, dpad=None):
    if button:
        gamepad.press_button(button=button)
    if dpad:
        gamepad.press_button(button=dpad)
    gamepad.update()
    time.sleep(0.2)
    if button:
        gamepad.release_button(button=button)
    if dpad:
        gamepad.release_button(button=dpad)
    gamepad.update()

def detect_action(keypoints):
    global facing_mode, last_toggle_time, last_action_time, hand_raise_state, combo2_triggered

    if keypoints.shape[0] < 17:
        return None

    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]

    now = time.time()
    shoulder_dist_x = abs(left_shoulder[0] - right_shoulder[0])
    wrist_dist_x = abs(left_wrist[0] - right_wrist[0])
    hip_center_x = (left_hip[0] + right_hip[0]) / 2
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2

    # Toggle facing
    if shoulder_dist_x < 60 and now - last_toggle_time > toggle_cooldown:
        facing_mode = 'left' if facing_mode == 'right' else 'right'
        last_toggle_time = now
        return f"Toggled Facing → {facing_mode.upper()}"

    # Combo 2 (hands high and crossed)
    if (left_wrist[1] < left_shoulder[1]) and (right_wrist[1] < right_shoulder[1]) and wrist_dist_x < 90:
        if not combo2_triggered and now - last_action_time > gesture_cooldown:
            press_combo(scorpion_combo_2[facing_mode])
            last_action_time = now
            combo2_triggered = True
            return f"Combo 2 ({facing_mode})"
    else:
        combo2_triggered = False

    # Combo 1 (both hands up and wide)
    if (left_wrist[1] < left_shoulder[1] - 40) and (right_wrist[1] < right_shoulder[1] - 40) and wrist_dist_x > 120:
        if now - last_action_time > gesture_cooldown:
            press_combo(scorpion_combo_1[facing_mode])
            last_action_time = now
            return f"Combo 1 ({facing_mode})"

    # Left Punch (X)
    if (left_elbow[0] - left_wrist[0]) > 60 and abs(left_wrist[1] - left_elbow[1]) < 60:
        if now - last_action_time > gesture_cooldown:
            press_and_release(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
            last_action_time = now
            return "Left Punch (X)"

    # Right Punch (Y)
    if (right_wrist[0] - right_elbow[0]) > 60 and abs(right_wrist[1] - right_elbow[1]) < 60:
        if now - last_action_time > gesture_cooldown:
            press_and_release(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
            last_action_time = now
            return "Right Punch (Y)"

    # B: Left hand above nose
    if left_wrist[1] < nose[1] - 40:
        if not hand_raise_state["left"]:
            gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            gamepad.update()
            hand_raise_state["left"] = True
            return "B (Left Hand Raised)"
    else:
        if hand_raise_state["left"]:
            gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            gamepad.update()
            hand_raise_state["left"] = False

    # A: Right hand above nose
    if right_wrist[1] < nose[1] - 40:
        if not hand_raise_state["right"]:
            gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            gamepad.update()
            hand_raise_state["right"] = True
            return "A (Right Hand Raised)"
    else:
        if hand_raise_state["right"]:
            gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            gamepad.update()
            hand_raise_state["right"] = False

    # Jump
    if nose[1] < shoulder_center_y - 80:
        press_and_release(dpad=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        return "Jump (UP)"

    # Crouch
    if nose[1] > shoulder_center_y + 80:
        press_and_release(dpad=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
        return "Crouch (DOWN)"

    # Move Left
    if nose[0] < hip_center_x - 60:
        press_and_release(dpad=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        return "Move Left"

    # Move Right
    if nose[0] > hip_center_x + 60:
        press_and_release(dpad=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        return "Move Right"

    return None

def draw_pose(frame, keypoints, action=None):
    for idx, (x, y) in enumerate(keypoints[:, :2]):
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(frame, str(idx), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for i, j in SKELETON_CONNECTIONS:
        pt1, pt2 = tuple(map(int, keypoints[i][:2])), tuple(map(int, keypoints[j][:2]))
        cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
    if action:
        cv2.putText(frame, f"Action: {action}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Facing: {facing_mode.upper()}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5)

    if results and results[0].keypoints is not None:
        poses = results[0].keypoints.data.cpu().numpy()
        for person in poses:
            if person.shape[0] < 17:
                continue
            action = detect_action(person)
            draw_pose(frame, person, action)
    else:
        cv2.putText(frame, "No person detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("MKXL Pose Controller", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
