# align_crop_dlib.py
import os
import dlib
import cv2

INPUT_DIR = "../train_data/part1"
OUTPUT_DIR = "../train_data/part1_cropped"
PREDICTOR_PATH = "../train_data/shape_predictor_5_face_landmarks.dat"
OUTPUT_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

for root, _, files in os.walk(INPUT_DIR):
    for fname in files:
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        in_path = os.path.join(root, fname)
        img = dlib.load_rgb_image(in_path)  # dlib uses RGB
        dets = detector(img, 1)
        if len(dets) == 0:
            continue

        # If there are multiple faces, take the largest / first one:
        det = max(dets, key=lambda r: r.width() * r.height())
        shape = predictor(img, det)
        # get_face_chip aligns the face given the shape -> returns numpy array
        face_chip = dlib.get_face_chip(img, shape, size=OUTPUT_SIZE)
        out_fname = os.path.join(OUTPUT_DIR, fname)
        face_bgr = cv2.cvtColor(face_chip, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_fname, face_bgr)
