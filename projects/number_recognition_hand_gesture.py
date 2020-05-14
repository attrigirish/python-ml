import cv2
import numpy as np
import math
video = cv2.VideoCapture(0)

while(1):
        ret,frame = video.read()
        frame=cv2.flip(frame,1)
        kernel=np.ones((3,3), np.uint8)

        #Region of Interest
        roi=frame[100:300, 100:300]

        cv2.rectangle(frame, (100,100),(300,300),(0,255,0), 0)
        hsv=cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        #Range of Skin Color in HSV
        lower_skin=np.array([0,20,70], dtype=np.uint8)
        upper_skin=np.array([20,255,255], dtype=np.uint8)

        #Extract skin color image
        mask=cv2.inRange(hsv,lower_skin,upper_skin)

        #Extrapolate the hand to fill dark spots within
        mask=cv2.dilate(mask,kernel,iterations = 4)

        #Blur the Image
        mask=cv2.GaussianBlur(mask,(5,5),100)
        
        #Find Contours
        contours, hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #Find contour of max area(hand)
        cnt=max(contours, key=lambda x: cv2.contourArea(x))

        #Approx the contour a little
        epsilon=0.0005*cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,epsilon,True)

        #Make Convex Hull around hand
        hull=cv2.convexHull(cnt)

        #Define area of hull around hand
        areahull=cv2.contourArea(hull)
        areacnt=cv2.contourArea(cnt)

        #Find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100

        #Find the defects in convex hull with respect to hand
        hull=cv2.convexHull(approx,returnPoints=False)
        defects=cv2.convexityDefects(approx,hull)

        #l = no of defects
        l=0

        #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d=defects[i,0]
            start=tuple(approx[s][0])
            end=tuple(approx[e][0])
            far=tuple(approx[f][0])
            pt=(100,180)

            #Find Length of all sides of triangle
            a=math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            b=math.sqrt((far[0]-start[0])**2 + (far[1]-start[1])**2)
            c=math.sqrt((end[0]-far[0])**2 + (end[1]-far[1])**2)
            s=(a+b+c)/2
            ar=math.sqrt(s*(s-a)*(s-b)*(s-c))

            #Distance between point and convex hull
            d=(2*ar)/a

            #apply cosine rule here
            angle=math.acos((b**2 + c**2 - a**2)/(2*b*c))*57

            #Ignore angles > 90 and ignore points very close to convex hull
            if angle<=90 and d>30:
                l+=1
                cv2.circle(roi, far, 3, [255,0,0], -1)

            #Draw lines around hand
            cv2.line(roi,start,end,[0,255,0],2)        
        l+=1

        #Print corresponding gestures which are in their ranges
        font=cv2.FONT_HERSHEY_SIMPLEX

        if(l==1):
            if(areacnt<2000):
                result="Put Hand in the box"
            else:
                if(arearatio<17.5):
                    result="0"
                else:
                    result="1"
        elif(l==2):
            result="2"
        elif(l==3):
            result="3"
        elif(l==4):
            result="4"
        elif(l==5):
            result="5"
        else:
            result="Try Again!"
                    
        cv2.putText(frame,result,(0,50),font,2,(0,0,255),3,cv2.LINE_AA)

        #Show the windows
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)        


        k=cv2.waitKey(25) & 0xFF
        if(k==27):
            break

cv2.destroyAllWindows()
video.release()
                  
