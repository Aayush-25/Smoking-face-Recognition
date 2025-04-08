import cv2
import numpy as np
import time
import threading
import pyttsx3
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector

class SmokingDetectionSystem:
    def __init__(self):
        # Initialize the video capture
        self.cap = cv2.VideoCapture(0)
        
        # Initialize the detectors
        self.face_detector = FaceMeshDetector(maxFaces=1)
        self.hand_detector = HandDetector(detectionCon=0.8, maxHands=2)
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Variables for detection logic
        self.smoking_detected = False
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds
        self.alert_in_progress = False
        
        # Detection parameters
        self.mouth_region_threshold = 100  # pixels, adjust based on testing
        
    def play_alert(self):
        """Function to play audio alert in a separate thread"""
        self.alert_in_progress = True
        self.engine.say("Smoking detected. Please stop smoking.")
        self.engine.runAndWait()
        self.alert_in_progress = False
        
    def detect_smoking(self, face_landmarks, hand_landmarks):
        """Detect if a person is smoking based on hand and face position"""
        if not face_landmarks or not hand_landmarks:
            return False
            
        # Get nose position (landmark index 1 in FaceMesh)
        nose = face_landmarks[1]
        
        # For each detected hand
        for hand in hand_landmarks:
            # Get index finger tip position (landmark 8)
            index_tip = hand["lmList"][8]
            
            # Calculate distance between nose and index finger tip
            distance = np.sqrt((nose[0] - index_tip[0])**2 + (nose[1] - index_tip[1])**2)
            
            # If the hand is close to the face (mouth region)
            if distance < self.mouth_region_threshold:
                return True
                
        return False
        
    def trigger_alert(self):
        """Trigger visual and audio alerts"""
        current_time = time.time()
        
        # Check if cooldown period has passed since last alert
        if current_time - self.last_alert_time > self.alert_cooldown and not self.alert_in_progress:
            # Start a new thread for audio alert
            alert_thread = threading.Thread(target=self.play_alert)
            alert_thread.daemon = True
            alert_thread.start()
            
            # Update last alert time
            self.last_alert_time = current_time
            
    def run(self):
        """Main loop for the smoking detection system"""
        while True:
            # Read frame from the webcam
            success, img = self.cap.read()
            if not success:
                print("Failed to grab frame")
                break
                
            # Flip the image horizontally for a more intuitive mirror view
            img = cv2.flip(img, 1)
            
            # Detect face mesh
            img, face_landmarks = self.face_detector.findFaceMesh(img, draw=False)
            
            # If face detected, get the first face landmarks
            face_landmarks = face_landmarks[0] if face_landmarks else []
            
            # Detect hands
            hands, img = self.hand_detector.findHands(img)
            
            # Check for smoking behavior
            if face_landmarks:
                self.smoking_detected = self.detect_smoking(face_landmarks, hands)
                
                # Draw nose position for debugging
                if len(face_landmarks) > 1:
                    cv2.circle(img, (face_landmarks[1][0], face_landmarks[1][1]), 5, (255, 0, 0), cv2.FILLED)
                    
                # Display smoking status
                status_text = "Status: Smoking Detected!" if self.smoking_detected else "Status: Normal"
                status_color = (0, 0, 255) if self.smoking_detected else (0, 255, 0)
                cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                
                # Trigger alert if smoking is detected
                if self.smoking_detected:
                    # Draw red rectangle for warning
                    cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)
                    cv2.putText(img, "SMOKING DETECTED!", (int(img.shape[1]/2) - 150, int(img.shape[0]/2)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
                    # Trigger the alert system
                    self.trigger_alert()
            
            # Display FPS
            fps_text = f"FPS: {int(1/(time.time() - (getattr(self, 'prev_time', time.time()-1/30))))}"
            self.prev_time = time.time()
            cv2.putText(img, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Display the resulting frame
            cv2.imshow('Smoking Detection System', img)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Release the webcam and close all windows
        self.cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    try:
        detector = SmokingDetectionSystem()
        detector.run()
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Error occurred: {str(e)}") 