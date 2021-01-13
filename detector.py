import cv2
cap = cv2.VideoCapture(0) #Give the name of it inplace of 0 for vedio


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while True:
    success, frame = cap.read() 
        
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),3)
    
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) == ord('q'): 
        break
            
        
cap.release() 
cv2.destroyAllWindows()