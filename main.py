import numpy as np
import mediapipe as mp
import cv2

pose_model = './pose_landmarker_full.task'
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

EXERCISE_CONFIG = {
    'squat': {
        'knee': [24, 26, 28], # HIP, KNEE, ANKLE (Right side)
        'hip': [12, 24, 26]   # SHOULDER, HIP, KNEE
    },
    'push-up': {
        'elbow': [12, 14, 16], # SHOULDER, ELBOW, WRIST
        'hip': [12, 24, 28]    # SHOULDER, HIP, ANKLE
    },
    'plank': {
        'shoulder': [24, 12, 14], # HIP, SHOULDER, ELBOW (Target: 80-100°)
        'hip': [12, 24, 28]       # SHOULDER, HIP, ANKLE (Target: 165-180°)
    }
}

COMPARING_ANGLES = {
    'squat': {
        'knee': {
            'correct_threshold': [60, 125],
            'error_msg': 'Tu sentadilla no es lo suficientemente profunda. Tu rodilla debe bajar cerca de los 90 grados.'
        },
        'hip': {
            'correct_threshold': [45, 180],
            'error_msg': 'Tu espalda está muy inclinada hacia adelante. Procura mantener el pecho erguido.'
        }
    },
    'push-up': {
        'elbow': {
            'correct_threshold': [70, 110],
            'error_msg': 'Tu flexión no es lo suficientemente profunda. Tu brazo debe doblarse hasta cerca de los 90 grados.'
        },
        'hip': {
            'correct_threshold': [155, 180],
            'error_msg': 'Tus hombros, espalda y cadera deben estar alineados. Evita elevar o dejar caer la cadera.'
        }
    },
    'plank': {
        'shoulder': {
            'correct_threshold': [80, 100],
            'error_msg': 'Mantén tus hombros directamente sobre tus codos.'
        },
        'hip': {
            'correct_threshold': [150, 180],
            'error_msg': 'Mantén tu cuerpo en una línea recta sin elevar ni bajar la cadera.'
        }
    }
}

pose_opt = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)


# Module 1
def extract_pose_vectors(video):
    video_landmarks_history = []

    with PoseLandmarker.create_from_options(pose_opt) as pose_landmarker:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

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

#module 2
def angle_calculation(history, exercise):
    angles_dict = {}
    joints_involved = EXERCISE_CONFIG[exercise]

    for frame in history:
        stamp = frame['timestamp_ms']
        landmarks = frame['landmarks']

        if not landmarks or len(landmarks) < 33:
            continue

        frame_angles = {}

        left_visibility = landmarks[11]['visibility'] + landmarks[23]['visibility']
        right_visibility = landmarks[12]['visibility'] + landmarks[24]['visibility']
        side_offset = 0 if right_visibility >= left_visibility else -1

        for current_joint, joints_ids in joints_involved.items():
            upper_id = joints_ids[0] + side_offset
            apex_id = joints_ids[1] + side_offset
            lower_id = joints_ids[2] + side_offset

            upper_point = landmarks[upper_id]
            apex_point = landmarks[apex_id]
            lower_point = landmarks[lower_id]

            upper = np.array([upper_point['x'], upper_point['y']])
            apex = np.array([apex_point['x'], apex_point['y']])
            lower = np.array([lower_point['x'], lower_point['y']])

            vector_u = upper - apex
            vector_v = lower - apex

            angle_u = np.arctan2(vector_u[1], vector_u[0])
            angle_v = np.arctan2(vector_v[1], vector_v[0])

            raw_angle = angle_v - angle_u
            angle_degrees = np.abs(raw_angle * 180.0 / np.pi)

            if angle_degrees > 180.0:
                angle_degrees = 360.0 - angle_degrees

            frame_angles[current_joint] = float(angle_degrees)

        angles_dict[stamp] = frame_angles

    return angles_dict


# Module 3
def is_exercise_active(joints, exercise):
    if exercise == 'squat':
        knee_angle = joints.get('knee', 180)
        return knee_angle <= 160

    elif exercise == 'push-up':
        elbow_angle = joints.get('elbow', 180)
        return elbow_angle <= 160

    elif exercise == 'plank':
        shoulder_angle = joints.get('shoulder', 180)
        return shoulder_angle <= 120

    return True


def extract_angle_ranges(data, exercise):

    ranges = {
        joint: {
            "min": float("inf"),
            "max": float("-inf")
        }
        for joint in EXERCISE_CONFIG[exercise].keys()
    }

    for ts, joints in data.items():

        if is_exercise_active(joints, exercise):

            for joint_name, angle in joints.items():

                if joint_name in ranges:

                    ranges[joint_name]["min"] = min(
                        ranges[joint_name]["min"],
                        angle
                    )

                    ranges[joint_name]["max"] = max(
                        ranges[joint_name]["max"],
                        angle
                    )

    return ranges

def posture_evaluation(data, exercise):
    evaluated_list = []
    comparing_data = COMPARING_ANGLES[exercise]

    for ts, joints in data.items():
        frame_eval = {'timestamp': ts}
        frame_eval.update(joints)

        frame_errors = []
        overall_status = 1

        active = is_exercise_active(joints, exercise)
        frame_eval['is_active'] = active
        
        if active:
            for joint_name, joint_angle in joints.items():
                if joint_name in comparing_data:
                    target_range = comparing_data[joint_name]['correct_threshold']
                    err_msg = comparing_data[joint_name]['error_msg']

                    if target_range[0] <= joint_angle <= target_range[1]:
                        frame_eval[f'{joint_name}_status'] = 1
                    else:
                        frame_eval[f'{joint_name}_status'] = 0
                        overall_status = 0
                        frame_errors.append(err_msg)

        else:
            overall_status = 1
            for joint_name in joints.keys():
                frame_eval[f'{joint_name}_status'] = 1
            

        frame_eval['evaluation'] = overall_status
        frame_eval['error_msg'] = frame_errors if frame_errors else None

        evaluated_list.append(frame_eval)

    return evaluated_list


if __name__ == '__main__':
    video_squat = cv2.VideoCapture('./videos/bad_plank.mp4')
    exercise = 'plank'


    print(f"Module 1 execution:\n")
    landmarks_history = extract_pose_vectors(video_squat)

    print(f"Module 2 execution:\n")
    angles = angle_calculation(landmarks_history, exercise)

    print(f"Module 3 execution:\n")
    evaluation = posture_evaluation(angles, exercise)
    for element in evaluation[:-5]:
        print(element)