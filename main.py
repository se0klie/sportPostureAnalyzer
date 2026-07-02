import numpy as np
import mediapipe as mp
import cv2
import time

pose_model = './pose_landmarker_full.task'
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

landmarks_needed = {
    'squat': [24, 26, 28], 
    'push-up': [12, 14, 16], 
    'plank': [12, 24, 26]
}

pose_opt = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)

video_squat = cv2.VideoCapture('./videos/squat.mp4')

def extract_pose_vectors(video):
    video_landmarks_history = []

    with PoseLandmarker.create_from_options(pose_opt) as pose_landmarker:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            timestamp_ms = int(video.get(cv2.CAP_PROP_POS_MSEC))
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            if pose_result.pose_landmarks:
                pose = pose_result.pose_landmarks[0]
                frame_vector = []
                
                for idx, landmark in enumerate(pose):
                    frame_vector.append({
                        'id': idx,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                video_landmarks_history.append({
                    'timestamp_ms': timestamp_ms,
                    'landmarks': frame_vector
                })

    video.release()
    return video_landmarks_history

def angle_calculation(history, exercise):
    angles_dict = {}
    
    joint_ids = landmarks_needed[exercise]
    upper_id, apex_id, lower_id = joint_ids[0], joint_ids[1], joint_ids[2]
    
    for frame in history:
        stamp = frame['timestamp_ms']
        landmarks = frame['landmarks']

        if not landmarks or len(landmarks) < 33:
            continue  

        upper_point = landmarks[upper_id] 
        apex_point  = landmarks[apex_id] 
        lower_point = landmarks[lower_id]

        upper = np.array([upper_point['x'], upper_point['y']])
        apex  = np.array([apex_point['x'], apex_point['y']])
        lower = np.array([lower_point['x'], lower_point['y']])

        vector_u = upper - apex
        vector_v = lower - apex

        angle_u = np.arctan2(vector_u[1], vector_u[0])
        angle_v = np.arctan2(vector_v[1], vector_v[0])

        raw_angle = angle_v - angle_u
        angle_degrees = np.abs(raw_angle * 180.0 / np.pi)

        if angle_degrees > 180.0:
            angle_degrees = 360.0 - angle_degrees

        angles_dict[stamp] = angle_degrees

    return angles_dict

landmarks_history = extract_pose_vectors(video_squat)
angles = angle_calculation(landmarks_history, 'squat')
print(angles)