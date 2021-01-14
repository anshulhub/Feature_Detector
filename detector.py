import cv2

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
faceCascade1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while True:
    success, frame = cap.read() 
            
        
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = faceCascade.detectMultiScale(imgGray, 1.05, 8)
    faces = faceCascade1.detectMultiScale(imgGray, 1.05, 8)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
    
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
    
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) == ord('q'): 
        break
            
        
cap.release() 
cv2.destroyAllWindows()