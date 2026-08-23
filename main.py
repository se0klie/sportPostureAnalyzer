import numpy as np
import mediapipe as mp
import cv2

POSE_MODEL_PATH = "./pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

DEPTH_JOINT = {
    "squat": "knee",
    "push-up": "elbow",
}

EXERCISE_CONFIG = {
    "squat": {
        "knee": [24, 26, 28],  # HIP, KNEE, ANKLE (lado derecho)
        "hip": [12, 24, 26],   # SHOULDER, HIP, KNEE
    },
    "push-up": {
        "elbow": [12, 14, 16],  # SHOULDER, ELBOW, WRIST
        "hip": [12, 24, 28],    # SHOULDER, HIP, ANKLE
    },
    "plank": {
        "shoulder": [24, 12, 14],  # HIP, SHOULDER, ELBOW
        "hip": [12, 24, 28],       # SHOULDER, HIP, ANKLE
    },
}

COMPARING_ANGLES = {
    "squat": {
        "knee": {
            "correct_threshold": [91, 125],
            "error_msg": "Tu sentadilla no es lo suficientemente profunda. Tu rodilla debe bajar cerca de los 90 grados.",
        },
        "hip": {
            "correct_threshold": [45, 180],
            "error_msg": "Tu espalda esta muy inclinada hacia adelante. Procura mantener el pecho erguido.",
        },
    },
    "push-up": {
        "elbow": {
            "correct_threshold": [70, 110],
            "error_msg": "Tu flexion no es lo suficientemente profunda. Tu brazo debe doblarse hasta cerca de los 90 grados.",
        },
        "hip": {
            "correct_threshold": [155, 180],
            "error_msg": "Tus hombros, espalda y cadera deben estar alineados. Evita elevar o dejar caer la cadera.",
        },
    },
    "plank": {
        "shoulder": {
            "correct_threshold": [80, 100],
            "error_msg": "Manten tus hombros directamente sobre tus codos.",
        },
        "hip": {
            "correct_threshold": [150, 180],
            "error_msg": "Manten tu cuerpo en una linea recta sin elevar ni bajar la cadera.",
        },
    },
}

BASE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
]

COLOR_OK = (110, 231, 110)      
COLOR_ERROR = (70, 70, 255)   
COLOR_INACTIVE = (150, 150, 150)  
COLOR_BASE = (90, 90, 90)      


def _build_pose_landmarker(model_path):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
    )
    return PoseLandmarker.create_from_options(options)

def extract_pose_vectors(video_path, model_path=POSE_MODEL_PATH, progress_cb=None):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0 
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    landmarks_history = []
    frame_idx = 0

    with _build_pose_landmarker(model_path) as pose_landmarker:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_for_detection = cv2.GaussianBlur(frame, (5, 5), 0)

            frame_rgb = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(frame_idx * 1000 / fps)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            if pose_result.pose_landmarks:
                pose = pose_result.pose_landmarks[0]
                frame_vector = [
                    {
                        "id": idx,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility,
                    }
                    for idx, landmark in enumerate(pose)
                ]

                landmarks_history.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "landmarks": frame_vector,
                    }
                )

            if progress_cb and frame_count:
                progress_cb(min(frame_idx / frame_count, 1.0) * 0.4)  # 0-40%

            frame_idx += 1

    video.release()
    meta = {"fps": fps, "width": width, "height": height, "frame_count": frame_count}
    return landmarks_history, meta

def angle_calculation(history, exercise, meta):
    angles_dict = {}
    joints_involved = EXERCISE_CONFIG[exercise]
    width, height = meta["width"], meta["height"]

    for frame in history:
        frame_idx = frame["frame_idx"]
        landmarks = frame["landmarks"]

        if not landmarks or len(landmarks) < 33:
            continue

        frame_angles = {"timestamp_ms": frame["timestamp_ms"]}

        left_visibility = landmarks[11]["visibility"] + landmarks[23]["visibility"]
        right_visibility = landmarks[12]["visibility"] + landmarks[24]["visibility"]
        side_offset = 0 if right_visibility >= left_visibility else -1

        for current_joint, joints_ids in joints_involved.items():
            upper_id = joints_ids[0] + side_offset
            apex_id = joints_ids[1] + side_offset
            lower_id = joints_ids[2] + side_offset

            upper_point = landmarks[upper_id]
            apex_point = landmarks[apex_id]
            lower_point = landmarks[lower_id]

            upper = np.array([upper_point["x"] * width, upper_point["y"] * height])
            apex = np.array([apex_point["x"] * width, apex_point["y"] * height])
            lower = np.array([lower_point["x"] * width, lower_point["y"] * height])

            vector_u = upper - apex
            vector_v = lower - apex

            angle_u = np.arctan2(vector_u[1], vector_u[0])
            angle_v = np.arctan2(vector_v[1], vector_v[0])

            raw_angle = angle_v - angle_u
            angle_degrees = np.abs(raw_angle * 180.0 / np.pi)

            if angle_degrees > 180.0:
                angle_degrees = 360.0 - angle_degrees

            frame_angles[current_joint] = float(angle_degrees)

        angles_dict[frame_idx] = frame_angles

    return angles_dict


def is_exercise_active(joints, exercise):
    if exercise == "squat":
        return joints.get("knee", 180) <= 160
    elif exercise == "push-up":
        return joints.get("elbow", 180) <= 160
    elif exercise == "plank":
        return joints.get("shoulder", 180) <= 120
    return True


def extract_angle_ranges(data, exercise):
    ranges = {
        joint: {"min": float("inf"), "max": float("-inf")}
        for joint in EXERCISE_CONFIG[exercise].keys()
    }

    for joints in data.values():
        if is_exercise_active(joints, exercise):
            for joint_name, angle in joints.items():
                if joint_name in ranges:
                    ranges[joint_name]["min"] = min(ranges[joint_name]["min"], angle)
                    ranges[joint_name]["max"] = max(ranges[joint_name]["max"], angle)

    return ranges


def segment_reps(data, exercise):
    reps = []
    current = []
    for frame_idx in sorted(data.keys()):
        joints = data[frame_idx]
        if is_exercise_active(joints, exercise):
            current.append(frame_idx)
        else:
            if current:
                reps.append(current)
                current = []
    if current:
        reps.append(current)
    return reps


def posture_evaluation(data, exercise):
    evaluated = {}
    comparing_data = COMPARING_ANGLES[exercise]
    depth_joint = DEPTH_JOINT.get(exercise)

    frame_depth_ok = {}
    if depth_joint:
        for rep in segment_reps(data, exercise):
            extremum = min(data[f][depth_joint] for f in rep) 
            lo, hi = comparing_data[depth_joint]["correct_threshold"]
            rep_ok = lo <= extremum <= hi
            for f in rep:
                frame_depth_ok[f] = rep_ok

    for frame_idx, joints in data.items():
        frame_eval = {"frame_idx": frame_idx, "timestamp_ms": joints["timestamp_ms"]}
        joint_angles = {k: v for k, v in joints.items() if k != "timestamp_ms"}
        frame_eval.update(joint_angles)

        frame_errors = []
        overall_status = 1
        active = is_exercise_active(joint_angles, exercise)
        frame_eval["is_active"] = active

        if active:
            for joint_name, joint_angle in joint_angles.items():
                if joint_name not in comparing_data:
                    continue

                if joint_name == depth_joint:
                    ok = frame_depth_ok.get(frame_idx, True)
                else:
                    lo, hi = comparing_data[joint_name]["correct_threshold"]
                    ok = lo <= joint_angle <= hi

                frame_eval[f"{joint_name}_status"] = 1 if ok else 0
                if not ok:
                    overall_status = 0
                    frame_errors.append(comparing_data[joint_name]["error_msg"])
        else:
            for joint_name in joint_angles.keys():
                frame_eval[f"{joint_name}_status"] = 1

        frame_eval["evaluation"] = overall_status
        frame_eval["error_msg"] = frame_errors if frame_errors else None

        evaluated[frame_idx] = frame_eval

    return evaluated

def _to_px(point, width, height):
    return int(point["x"] * width), int(point["y"] * height)


def _draw_base_skeleton(frame, landmarks, width, height):
    for a, b in BASE_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            pa = _to_px(landmarks[a], width, height)
            pb = _to_px(landmarks[b], width, height)
            cv2.line(frame, pa, pb, COLOR_BASE, 2, cv2.LINE_AA)


def _draw_joint_segment(frame, landmarks, joint_ids, side_offset, color, width, height, label=None):
    upper_id = joint_ids[0] + side_offset
    apex_id = joint_ids[1] + side_offset
    lower_id = joint_ids[2] + side_offset
    if max(upper_id, apex_id, lower_id) >= len(landmarks):
        return

    upper = _to_px(landmarks[upper_id], width, height)
    apex = _to_px(landmarks[apex_id], width, height)
    lower = _to_px(landmarks[lower_id], width, height)

    cv2.line(frame, upper, apex, color, 5, cv2.LINE_AA)
    cv2.line(frame, apex, lower, color, 5, cv2.LINE_AA)
    cv2.circle(frame, apex, 7, color, -1, cv2.LINE_AA)

    if label is not None:
        cv2.putText(
            frame, label, (apex[0] + 10, apex[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )


def _draw_message_banner(frame, messages, width, height, active):
    if not active:
        text = "Fase de descanso / preparacion"
        color = COLOR_INACTIVE
        lines = [text]
    elif not messages:
        lines = ["Postura correcta"]
        color = COLOR_OK
    else:
        lines = messages
        color = COLOR_ERROR

    overlay = frame.copy()
    banner_h = 34 + 26 * len(lines)
    cv2.rectangle(overlay, (0, height - banner_h), (width, height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = height - banner_h + 26
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 26


def render_overlay_video(video_path, landmarks_history, evaluation, exercise, meta, output_path, progress_cb=None):
    joints_involved = EXERCISE_CONFIG[exercise]
    landmarks_by_frame = {h["frame_idx"]: h["landmarks"] for h in landmarks_history}

    width, height, fps = meta["width"], meta["height"], meta["fps"]

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    video = cv2.VideoCapture(video_path)
    frame_idx = 0
    frame_count = meta.get("frame_count") or 1

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) 

        landmarks = landmarks_by_frame.get(frame_idx)
        frame_eval = evaluation.get(frame_idx)

        if landmarks and frame_eval:
            _draw_base_skeleton(frame, landmarks, width, height)

            left_visibility = landmarks[11]["visibility"] + landmarks[23]["visibility"]
            right_visibility = landmarks[12]["visibility"] + landmarks[24]["visibility"]
            side_offset = 0 if right_visibility >= left_visibility else -1

            active = frame_eval["is_active"]
            for joint_name, joint_ids in joints_involved.items():
                status = frame_eval.get(f"{joint_name}_status", 1)
                if not active:
                    color = COLOR_INACTIVE
                elif status == 1:
                    color = COLOR_OK
                else:
                    color = COLOR_ERROR
                angle_val = frame_eval.get(joint_name)
                label = f"{joint_name} {angle_val:.0f} deg" if angle_val is not None else None
                _draw_joint_segment(frame, landmarks, joint_ids, side_offset, color, width, height, label)

            _draw_message_banner(frame, frame_eval.get("error_msg"), width, height, active)
        else:
            cv2.putText(
                frame, "Persona no detectada", (16, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INACTIVE, 2, cv2.LINE_AA,
            )

        writer.write(frame)

        if progress_cb and frame_count:
            progress_cb(0.6 + min(frame_idx / frame_count, 1.0) * 0.4)  # 60-100%

        frame_idx += 1

    video.release()
    writer.release()
    return output_path


def summarize(evaluation, exercise):
    total = len(evaluation)
    active_frames = [e for e in evaluation.values() if e["is_active"]]
    active_count = len(active_frames)
    correct_count = sum(1 for e in active_frames if e["evaluation"] == 1)

    per_joint = {}
    for joint in EXERCISE_CONFIG[exercise].keys():
        joint_frames = [e for e in active_frames if f"{joint}_status" in e]
        ok = sum(1 for e in joint_frames if e[f"{joint}_status"] == 1)
        per_joint[joint] = {
            "ok_pct": round(100 * ok / len(joint_frames), 1) if joint_frames else None,
        }

    message_counts = {}
    for e in active_frames:
        for msg in (e.get("error_msg") or []):
            message_counts[msg] = message_counts.get(msg, 0) + 1

    top_messages = sorted(message_counts.items(), key=lambda kv: -kv[1])

    return {
        "total_frames": total,
        "active_frames": active_count,
        "correct_frames": correct_count,
        "score_pct": round(100 * correct_count / active_count, 1) if active_count else None,
        "per_joint": per_joint,
        "top_messages": [{"message": m, "count": c} for m, c in top_messages],
    }


def process_video(video_path, exercise, output_path, model_path=POSE_MODEL_PATH, progress_cb=None):
    if exercise not in EXERCISE_CONFIG:
        raise ValueError(f"Ejercicio no soportado: {exercise}")

    landmarks_history, meta = extract_pose_vectors(video_path, model_path, progress_cb)
    if not landmarks_history:
        raise ValueError("No se detecto a ninguna persona en el video.")

    angles = angle_calculation(landmarks_history, exercise, meta)
    evaluation = posture_evaluation(angles, exercise)
    ranges = extract_angle_ranges(
        {k: {kk: vv for kk, vv in v.items() if kk != "timestamp_ms"} for k, v in angles.items()},
        exercise,
    )

    render_overlay_video(video_path, landmarks_history, evaluation, exercise, meta, output_path, progress_cb)

    summary = summarize(evaluation, exercise)
    summary["ranges"] = ranges
    return output_path, summary