import cv2
import mediapipe as mp
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Game settings
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
NUM_POSITIONS = 4
SHOW_TIME = 5
COUNTDOWN_TIME = 3
LIVES = 3
CIRCLE_RADIUS = 40
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Initialize game variables
sequence = []
current_number = 1
score = 0
lives = LIVES
game_state = "SHOW"
start_time = 0
last_detection_time = 0
DETECTION_COOLDOWN = 0.5
correct_guesses = set()

# Create positions for numbers
positions = [(int(SCREEN_WIDTH * (i + 1) / (NUM_POSITIONS + 1)), SCREEN_HEIGHT // 2) for i in range(NUM_POSITIONS)]

def generate_sequence():
    return random.sample(range(1, NUM_POSITIONS + 1), NUM_POSITIONS)

def draw_circles(frame, numbers=None):
    for i, pos in enumerate(positions):
        if i in correct_guesses:
            cv2.circle(frame, pos, CIRCLE_RADIUS, (0, 255, 0), -1)
            cv2.putText(frame, str(sequence[i]), (pos[0] - 12, pos[1] + 12), FONT, 1, (255, 255, 255), 2)
        elif numbers and game_state == "SHOW":
            cv2.circle(frame, pos, CIRCLE_RADIUS, (0, 0, 255), -1)
            cv2.putText(frame, str(numbers[i]), (pos[0] - 12, pos[1] + 12), FONT, 1, (255, 255, 255), 2)
        else:
            cv2.circle(frame, pos, CIRCLE_RADIUS, (0, 255, 0), 2)

def get_pointed_position(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    x, y = int(index_tip.x * SCREEN_WIDTH), int(index_tip.y * SCREEN_HEIGHT)
    
    for i, pos in enumerate(positions):
        if ((x - pos[0])**2 + (y - pos[1])**2)**0.5 < CIRCLE_RADIUS:
            return i
    return None

# Main game loop
cap = cv2.VideoCapture(0)
cap.set(3, SCREEN_WIDTH)
cap.set(4, SCREEN_HEIGHT)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if game_state == "SHOW":
        if not sequence:
            sequence = generate_sequence()
            start_time = time.time()
            correct_guesses.clear()
        draw_circles(frame, sequence)
        remaining_time = int(SHOW_TIME - (time.time() - start_time))
        cv2.putText(frame, f"Remember the positions! {remaining_time}s", (10, 30), FONT, 0.7, (255, 255, 255), 2)
        if time.time() - start_time > SHOW_TIME:
            game_state = "PLAY"
            current_number = 1

    elif game_state == "PLAY":
        draw_circles(frame)
        cv2.putText(frame, f"Point to {current_number}", (10, 30), FONT, 0.7, (255, 255, 255), 2)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            pointed_pos = get_pointed_position(hand_landmarks)

            current_time = time.time()
            if pointed_pos is not None and current_time - last_detection_time > DETECTION_COOLDOWN:
                last_detection_time = current_time
                
                # Check if the number is already revealed
                if pointed_pos not in correct_guesses:
                    if sequence[pointed_pos] == current_number:
                        correct_guesses.add(pointed_pos)
                        current_number += 1
                        if current_number > NUM_POSITIONS:
                            score += 1
                            game_state = "COUNTDOWN"
                            start_time = time.time()
                    else:
                        lives -= 1
                        if lives == 0:
                            game_state = "GAMEOVER"
                        else:
                            game_state = "SHOW"
                            sequence = []

    elif game_state == "COUNTDOWN":
        draw_circles(frame)
        remaining_time = int(COUNTDOWN_TIME - (time.time() - start_time))
        cv2.putText(frame, f"Great job! Next round in {remaining_time}s", (10, 30), FONT, 0.7, (255, 255, 255), 2)
        if time.time() - start_time > COUNTDOWN_TIME:
            game_state = "SHOW"
            sequence = []

    elif game_state == "GAMEOVER":
        cv2.putText(frame, "GAME OVER", (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2), FONT, 2, (0, 0, 255), 3)
        cv2.putText(frame, f"Final Score: {score}", (SCREEN_WIDTH // 3, SCREEN_HEIGHT // 2 + 50), FONT, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'R' to restart", (SCREEN_WIDTH // 3, SCREEN_HEIGHT // 2 + 100), FONT, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Score: {score} Lives: {lives}", (10, SCREEN_HEIGHT - 10), FONT, 0.5, (255, 255, 255), 1)
    cv2.imshow('Memory Game', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and game_state == "GAMEOVER":
        score, lives, game_state, sequence = 0, LIVES, "SHOW", []
        correct_guesses.clear()

hands.close()
cap.release()
cv2.destroyAllWindows()
