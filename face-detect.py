import cv2
import datetime

face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade-eye.xml')
cap = cv2.VideoCapture(0) #you can put any video too!
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) != 0:# check if there is a Face
        # cascading Faces
        for (x, y, w, h) in faces:
            center = (x + w//2, y + h//2)
            radius = (min(w, h) // 2) + 10
            cv2.circle(img, center, radius, (0, 0, 255), 2)
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            font_size = min(w/10,h/10)/20
            cv2.putText(img,current_datetime,(int(x-(font_size*100)),y+4),cv2.FONT_HERSHEY_COMPLEX,font_size,(255,217,0),2)
            cv2.putText(img,"Face",(x,h+y),cv2.FONT_HERSHEY_COMPLEX,font_size,(255,217,0),2)
            # cascading eyes
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 81, 255), 2)
                font_size2 = min(ew/5,eh/5)/20
                cv2.putText(img,"Eye",(ex+x,ey+y),cv2.FONT_HERSHEY_COMPLEX,font_size2,(255,217,0),2)
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
    else:
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
