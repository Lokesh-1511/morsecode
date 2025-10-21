import cv2
import mediapipe as mp
import time

# ------------------------------
# Morse code dictionary
# ------------------------------
MORSE_CODE_DICT = {
    '.-':'A', '-...':'B', '-.-.':'C', '-..':'D', '.':'E', '..-.':'F',
    '--.':'G', '....':'H', '..':'I', '.---':'J', '-.-':'K', '.-..':'L',
    '--':'M', '-.':'N', '---':'O', '.--.':'P', '--.-':'Q', '.-.':'R',
    '...':'S', '-':'T', '..-':'U', '...-':'V', '.--':'W', '-..-':'X',
    '-.--':'Y', '--..':'Z', '-----':'0', '.----':'1', '..---':'2',
    '...--':'3', '....-':'4', '.....':'5', '-....':'6', '--...':'7',
    '---..':'8', '----.':'9'
}

# ------------------------------
# Initialize Mediapipe Face Mesh
# ------------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

# Eye landmark indices
EYE_TOP = 159
EYE_BOTTOM = 145
EYE_LEFT = 33
EYE_RIGHT = 133

# Blink thresholds
DOT_DURATION = 0.3
DASH_DURATION = 0.7
LETTER_GAP = 1.0  # seconds between letters

# ------------------------------
# Variables
# ------------------------------
blink_start = None
last_blink_time = time.time()
morse_sequence = ''
decoded_message = ''

# ------------------------------
# Eye Aspect Ratio
# ------------------------------
def eye_aspect_ratio(landmarks, img_h, img_w):
    top = landmarks[EYE_TOP]
    bottom = landmarks[EYE_BOTTOM]
    left = landmarks[EYE_LEFT]
    right = landmarks[EYE_RIGHT]
    # Convert normalized to pixel coordinates
    t = (int(top.x*img_w), int(top.y*img_h))
    b = (int(bottom.x*img_w), int(bottom.y*img_h))
    l = (int(left.x*img_w), int(left.y*img_h))
    r = (int(right.x*img_w), int(right.y*img_h))
    # Vertical / horizontal ratio
    ver = ((t[1]-b[1])**2 + (t[0]-b[0])**2)**0.5
    hor = ((l[1]-r[1])**2 + (l[0]-r[0])**2)**0.5
    return ver / hor

# ------------------------------
# Webcam capture
# ------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame,1)
    img_h, img_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        ear = eye_aspect_ratio(landmarks, img_h, img_w)

        # Detect blink
        if ear < 0.25:
            if blink_start is None:
                blink_start = time.time()
        else:
            if blink_start is not None:
                blink_duration = time.time() - blink_start
                blink_start = None
                last_blink_time = time.time()

                if blink_duration < DOT_DURATION:
                    morse_sequence += '.'
                elif blink_duration < DASH_DURATION:
                    morse_sequence += '-'

        # Letter separation
        if time.time() - last_blink_time > LETTER_GAP and morse_sequence:
            decoded_message += MORSE_CODE_DICT.get(morse_sequence, '?')
            morse_sequence = ''

    # Display
    cv2.putText(frame, f"Message: {decoded_message}", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Morse Eye Blinks", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()