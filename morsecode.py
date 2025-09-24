import cv2
import time
from collections import deque

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
# Load Haar cascades
# ------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ------------------------------
# Blink detection parameters
# ------------------------------
DOT_DURATION   = 2.0
DASH_DURATION  = 3.0
LETTER_GAP     = 2.0  # seconds between letters
WORD_GAP       = 3.0    # seconds between words
FRAME_BUFFER   = 3      # consecutive frames for smoothing

blink_start    = None
last_blink_time = time.time()
last_word_time  = time.time()
morse_sequence  = ''
decoded_message = ''

# Frame history for smoothing
eye_history = deque(maxlen=FRAME_BUFFER)

# ------------------------------
# Webcam
# ------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # ------------------------------
    # Preprocess for lighting robustness
    # ------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)                         # normalize brightness
    gray = cv2.GaussianBlur(gray, (5,5), 0)              # reduce noise
    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=30)  # adjust contrast/brightness

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes_detected = False

    for (x,y,w,h) in faces:
        roi_gray  = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) > 0:
            eyes_detected = True
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

    # ------------------------------
    # Frame smoothing
    # ------------------------------
    eye_history.append(eyes_detected)
    eyes_present = sum(eye_history) > FRAME_BUFFER//2  # majority vote

    # ------------------------------
    # Blink detection
    # ------------------------------
    current_time = time.time()
    if not eyes_present:
        if blink_start is None:
            blink_start = current_time
    else:
        if blink_start is not None:
            blink_duration = current_time - blink_start
            blink_start = None
            last_blink_time = current_time
            last_word_time = current_time

            if blink_duration < DOT_DURATION:
                morse_sequence += '.'
            elif blink_duration < DASH_DURATION:
                morse_sequence += '-'

    # ------------------------------
    # Letter separation
    # ------------------------------
    if morse_sequence and (current_time - last_blink_time) > LETTER_GAP:
        decoded_message += MORSE_CODE_DICT.get(morse_sequence, '?')
        morse_sequence = ''

    # ------------------------------
    # Word separation
    # ------------------------------
    if decoded_message and (current_time - last_word_time) > WORD_GAP:
        decoded_message += ' '
        last_word_time = current_time

    # ------------------------------
    # Display message
    # ------------------------------
    cv2.putText(frame, f"Message: {decoded_message}", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Morse Eye Blinks", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()