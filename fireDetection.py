import cv2         # Library for openCV
import threading   
import playsound   
import smtplib     

fire_cascade = cv2.CascadeClassifier('fire_detection_cascade_model.xml') # To access xml file which includes positive and negative images of fire. (Trained images)
                                                                         # File is also provided with the code.

vid = cv2.VideoCapture(0) #cam -0 for pc, 1 for usb connected
runOnce = False 

def play_alarm_sound_function(): 
    playsound.playsound('Alarm Sound.mp3',True) 
    print("Fire alarm end") 
 
def send_mail_function(): 
    
    recipientmail = "priyankachowdary3309@gmail.com" # recipients mail
    recipientmail = recipientmail.lower() 
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo() 
        server.starttls()
        server.login("priyanka_vattikuti@srmap.edu.in", 'Priya@1623') # Senders mail ID and password
        server.sendmail('priyankachowdary3309@gmail.com', recipientmail, "Warning fire accident has been reported!!!") # recipients mail with mail message
        print("Alert mail sent sucesfully to {}".format(recipientmail)) 
        server.close() ## To close server
        
    except Exception as e:
        print(e)
		
while(True):
    Alarm_Status = False
    ret, frame = vid.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # To convert frame into gray color
    fire = fire_cascade.detectMultiScale(frame, 1.2, 5) # to provide frame resolution

    ## to highlight fire with square 
    for (x,y,w,h) in fire:
        cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        print("Fire alarm initiated")
        threading.Thread(target=play_alarm_sound_function).start()  

        if runOnce == False:
            print("Mail send initiated")
            threading.Thread(target=send_mail_function).start() # To call alarm thread
            runOnce = True
        if runOnce == True:
            print("Mail is already sent once")
            runOnce = True

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
