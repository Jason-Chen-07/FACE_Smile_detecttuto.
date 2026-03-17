import cv2
import time

# Load Haar cascade classifiers using OpenCV's built-in path
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)

# Check if classifiers are loaded correctly
if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
    print("Error: Failed to load cascade classifiers")
    exit()

# Initialize webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Check if camera is accessible
if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

# Used to control screenshot interval
last_capture_time = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (required for detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Define region of interest (ROI) for eyes and smile detection
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Detect smile within face region
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

            # Display "Smiling!" text above face
            cv2.putText(frame, "Smiling!", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Take a screenshot if smiling (with 2-second cooldown)
            if time.time() - last_capture_time > 2:
                filename = f"smile_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"Captured: {filename}")
                last_capture_time = time.time()

    # Show the result in a window
    cv2.imshow('Face Eye Smile Detection', frame)

    # Press ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
