import cv2  # type: ignore
face_cascade = cv2.CascadeClassifier("opencv\\face&object_detection(advance_pro)level\\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("opencv\\face&object_detection(advance_pro)level\\haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("opencv\\face&object_detection(advance_pro)level\\haarcascade_smile.xml")
cap =cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces= face_cascade.detectMultiScale(gray,1, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0) , 2)

    region_gray = gray[y:y+h, x:x+w]
    region_color = frame[y:y+h, x:x+w]

    eyes= eye_cascade.detectMultiScale(region_gray,1.1, 5)
    if len(eyes) > 0:
        cv2.putText(region_color, "Eyes Detected", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0),2)
        
    smiles= smile_cascade.detectMultiScale(region_gray,1.5, 20)
    if len(smiles) > 0:
        cv2.putText(region_color, "Smiles Detected", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

    cv2.imshow("Face Eye Smile Detection", frame)
    
    if cv2.waitKey(1) & 0xFF== ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

#project done for face , eye , smile detection using haarcascade files and opencv library.