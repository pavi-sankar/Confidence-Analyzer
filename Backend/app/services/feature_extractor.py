import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import math
from scipy.spatial import distance
import tempfile
import os
from typing import List, Dict, Any

class VideoFeatureExtractor:
    def __init__(self):
        # Initialize MediaPipe (same as your existing code)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
    
    def extract_features_from_video_bytes(self, video_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract features from video bytes in memory"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name
        
        try:
            return self.extract_features_from_video_file(temp_path)
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    def extract_features_from_video_file(self, video_path: str) -> List[Dict[str, Any]]:
        """Extract features from video file path"""
        cap = cv2.VideoCapture(video_path)
        features_list = []
        face_history = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame to reduce computation (adjust as needed)
            if frame_count % 5 == 0:
                frame_features = self.process_frame(frame, face_history, frame_count)
                if frame_features:
                    features_list.append(frame_features)
            
            frame_count += 1
        
        cap.release()
        return features_list
    
    def process_frame(self, frame, face_history, frame_number):
        """Process a single frame"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = frame.shape
        
        # Process with MediaPipe
        face_results = self.face_mesh.process(img_rgb)
        pose_results = self.pose.process(img_rgb)
        hand_results = self.hands.process(img_rgb)
        
        features = {
            'frame_number': frame_number,
            'img_width': img_w,
            'img_height': img_h
        }
        
        # Face analysis
        if face_results.multi_face_landmarks:
            face_lm = face_results.multi_face_landmarks[0]
            face_history.append(face_lm)
            
            # Eye contact analysis
            eye_contact, ear = self.analyze_eye_contact(face_lm, img_w, img_h)
            features['eye_contact'] = eye_contact
            features['eye_aspect_ratio'] = ear
            
            # Face touching detection
            face_touch = self.detect_face_touching(
                face_lm, 
                hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else None,
                img_w, img_h
            )
            features['face_touching'] = face_touch
            
            # Smile analysis
            smile, smile_ratio = self.analyze_smile(face_lm)
            features['genuine_smile'] = smile
            features['smile_ratio'] = smile_ratio
            
            # Head pose analysis
            head_upright, yaw, pitch = self.analyze_head_pose(face_lm)
            features['head_upright'] = head_upright
            features['head_yaw'] = yaw
            features['head_pitch'] = pitch
            
            # Head nod detection
            nod = self.detect_head_nod(face_history)
            features['head_nod'] = nod
            
            # Brow movement analysis
            brow, brow_dist = self.analyze_brow_movement(face_lm)
            features['brow_raised'] = brow
            features['brow_distance'] = brow_dist
            
            # Jaw tension analysis
            jaw_relaxed, jaw_open = self.analyze_jaw_tension(face_lm)
            features['jaw_relaxed'] = jaw_relaxed
            features['jaw_openness'] = jaw_open
            
        else:
            # Face not detected - set default values
            for key in ['eye_contact', 'eye_aspect_ratio', 'face_touching', 'genuine_smile',
                       'smile_ratio', 'head_upright', 'head_yaw', 'head_pitch', 'head_nod',
                       'brow_raised', 'brow_distance', 'jaw_relaxed', 'jaw_openness']:
                features[key] = -1
        
        # Pose analysis
        if pose_results.pose_landmarks:
            pose_lm = pose_results.pose_landmarks
            
            # Arm position analysis
            arms_open, midline = self.analyze_arm_position(pose_lm)
            features['arms_uncrossed'] = arms_open
            
            # Posture analysis
            upright, open_posture, shoulder_w, torso_angle = self.analyze_posture(pose_lm)
            features['upright_torso'] = upright
            features['open_posture'] = open_posture
            features['shoulder_width'] = shoulder_w
            features['torso_angle'] = torso_angle
        else:
            features['arms_uncrossed'] = -1
            features['upright_torso'] = -1
            features['open_posture'] = -1
            features['shoulder_width'] = 0
            features['torso_angle'] = 0
        
        # Hand gestures analysis
        expressive, movement = self.analyze_hand_gestures(
            hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else None
        )
        features['hand_gestures_expressive'] = expressive
        features['hand_movement'] = movement
        
        return features

    # ===== COPY ALL YOUR EXISTING ANALYSIS METHODS HERE =====
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio for eye contact detection"""
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def analyze_eye_contact(self, face_landmarks, img_w, img_h):
        """Analyze eye contact and gaze direction"""
        left_eye = [(face_landmarks.landmark[i].x * img_w, 
                     face_landmarks.landmark[i].y * img_h) 
                    for i in [33, 160, 158, 133, 153, 144]]
        right_eye = [(face_landmarks.landmark[i].x * img_w, 
                      face_landmarks.landmark[i].y * img_h) 
                     for i in [362, 385, 387, 263, 373, 380]]
        
        left_ear = self.calculate_eye_aspect_ratio(left_eye)
        right_ear = self.calculate_eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2
        
        eye_contact_score = 1 if avg_ear > 0.2 else 0
        return eye_contact_score, avg_ear
    
    def detect_face_touching(self, face_landmarks, hand_landmarks, img_w, img_h):
        """Detect if hands are touching face (nervousness indicator)"""
        if hand_landmarks is None or len(hand_landmarks) == 0:
            return 0
        
        # Get face center and boundary
        face_points = [(lm.x * img_w, lm.y * img_h) for lm in face_landmarks.landmark]
        face_center_x = np.mean([p[0] for p in face_points])
        face_center_y = np.mean([p[1] for p in face_points])
        
        # Check if any hand is near face
        for hand in hand_landmarks:
            for landmark in hand.landmark:
                hand_x = landmark.x * img_w
                hand_y = landmark.y * img_h
                dist = math.sqrt((hand_x - face_center_x)**2 + (hand_y - face_center_y)**2)
                
                # If hand is within face region
                if dist < img_w * 0.15:
                    return 1
        return 0
    
    def analyze_smile(self, face_landmarks):
        """Analyze smile (genuine vs forced)"""
        # Mouth corners: 61 (left), 291 (right)
        # Upper lip: 13, Lower lip: 14
        mouth_left = face_landmarks.landmark[61]
        mouth_right = face_landmarks.landmark[291]
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        
        # Calculate mouth width and openness
        mouth_width = math.sqrt(
            (mouth_right.x - mouth_left.x)**2 + 
            (mouth_right.y - mouth_left.y)**2
        )
        mouth_height = abs(upper_lip.y - lower_lip.y)
        
        smile_ratio = mouth_width / (mouth_height + 0.001)
        
        # Check eye involvement (Duchenne smile)
        left_eye_y = face_landmarks.landmark[159].y
        right_eye_y = face_landmarks.landmark[386].y
        avg_eye_y = (left_eye_y + right_eye_y) / 2
        
        # Genuine smile if eyes are slightly squinted and mouth is wide
        genuine_smile = 1 if smile_ratio > 15 and mouth_width > 0.15 else 0
        return genuine_smile, smile_ratio
    
    def analyze_head_pose(self, face_landmarks):
        """Analyze head pose (pitch, yaw, roll)"""
        # Key points for head pose
        nose_tip = face_landmarks.landmark[1]
        chin = face_landmarks.landmark[152]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        
        # Calculate yaw (left-right rotation)
        eye_center_x = (left_eye.x + right_eye.x) / 2
        yaw = nose_tip.x - eye_center_x
        
        # Calculate pitch (up-down tilt)
        pitch = nose_tip.y - chin.y
        
        # Upright and forward facing = confident
        upright_score = 1 if abs(yaw) < 0.05 and pitch < 0.15 else 0
        return upright_score, yaw, pitch
    
    def detect_head_nod(self, face_landmarks_history):
        """Detect head nodding from frame history"""
        if len(face_landmarks_history) < 5:
            return 0
        
        # Track nose tip vertical movement
        nose_positions = [lm.landmark[1].y for lm in face_landmarks_history[-5:]]
        movement = np.std(nose_positions)
        
        # Head nod detected if there's rhythmic vertical movement
        nod_detected = 1 if movement > 0.01 else 0
        return nod_detected
    
    def analyze_brow_movement(self, face_landmarks):
        """Analyze eyebrow movements"""
        # Brow landmarks: left (70, 63), right (300, 293)
        left_brow = face_landmarks.landmark[70]
        right_brow = face_landmarks.landmark[300]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        
        # Distance between brow and eye
        left_dist = abs(left_brow.y - left_eye.y)
        right_dist = abs(right_brow.y - right_eye.y)
        avg_dist = (left_dist + right_dist) / 2
        
        # Raised brows (confident) vs furrowed (uncertain)
        brow_score = 1 if avg_dist > 0.04 else 0
        return brow_score, avg_dist
    
    def analyze_jaw_tension(self, face_landmarks):
        """Analyze jaw tension"""
        # Jaw points: 172 (bottom), 152 (chin)
        jaw_bottom = face_landmarks.landmark[172]
        chin = face_landmarks.landmark[152]
        mouth_top = face_landmarks.landmark[13]
        
        jaw_openness = abs(jaw_bottom.y - mouth_top.y)
        
        # Relaxed jaw = confident
        relaxed_score = 1 if jaw_openness > 0.05 else 0
        return relaxed_score, jaw_openness
    
    def analyze_arm_position(self, pose_landmarks):
        """Analyze if arms are crossed or open"""
        if pose_landmarks is None:
            return -1, 0  # Not visible
        
        # Shoulders: 11 (left), 12 (right)
        # Elbows: 13 (left), 14 (right)
        # Wrists: 15 (left), 16 (right)
        left_shoulder = pose_landmarks.landmark[11]
        right_shoulder = pose_landmarks.landmark[12]
        left_wrist = pose_landmarks.landmark[15]
        right_wrist = pose_landmarks.landmark[16]
        
        # Check visibility
        if (left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5 or
            left_wrist.visibility < 0.5 or right_wrist.visibility < 0.5):
            return -1, 0
        
        # Check if arms are crossed (wrists cross midline)
        midline_x = (left_shoulder.x + right_shoulder.x) / 2
        left_crossed = left_wrist.x > midline_x
        right_crossed = right_wrist.x < midline_x
        
        arms_open = 0 if (left_crossed and right_crossed) else 1
        return arms_open, midline_x
    
    def analyze_hand_gestures(self, hand_landmarks):
        """Analyze hand gesture expressiveness"""
        if hand_landmarks is None or len(hand_landmarks) == 0:
            return 0, 0
        
        total_movement = 0
        for hand in hand_landmarks:
            # Calculate spread of fingers
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
            
            # Measure variance in hand position (expressiveness)
            x_var = np.var([lm[0] for lm in landmarks])
            y_var = np.var([lm[1] for lm in landmarks])
            movement = x_var + y_var
            total_movement += movement
        
        # Expressive gestures = confident
        expressive_score = 1 if total_movement > 0.01 else 0
        return expressive_score, total_movement
    
    def analyze_posture(self, pose_landmarks):
        """Analyze torso posture and openness"""
        if pose_landmarks is None:
            return -1, -1, 0, 0
        
        # Shoulders: 11 (left), 12 (right)
        # Hips: 23 (left), 24 (right)
        left_shoulder = pose_landmarks.landmark[11]
        right_shoulder = pose_landmarks.landmark[12]
        left_hip = pose_landmarks.landmark[23]
        right_hip = pose_landmarks.landmark[24]
        
        # Check visibility
        shoulder_vis = (left_shoulder.visibility + right_shoulder.visibility) / 2
        hip_vis = (left_hip.visibility + right_hip.visibility) / 2
        
        if shoulder_vis < 0.5:
            return -1, -1, 0, 0
        
        # Calculate uprightness
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        if hip_vis > 0.5:
            hip_center_y = (left_hip.y + right_hip.y) / 2
            torso_angle = shoulder_center_y - hip_center_y
            upright_posture = 1 if torso_angle < -0.3 else 0
        else:
            upright_posture = -1
            torso_angle = 0
        
        # Calculate shoulder openness
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        open_posture = 1 if shoulder_width > 0.25 else 0
        
        return upright_posture, open_posture, shoulder_width, torso_angle

# Global instance
feature_extractor = VideoFeatureExtractor()