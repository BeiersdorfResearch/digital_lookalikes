import pandas as pd
import os
from pathlib import Path
import cv2
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
import time

# Set up logging
logging.basicConfig(filename='landmark_processing.log', level=logging.DEBUG)

base_options = python.BaseOptions(model_asset_path = cfg.landmarks.model_path)
face_landmarker = vision.FaceLandmarker
face_landmarker_options = vision.FaceLandmarkerOptions
options = face_landmarker_options(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)

def list_files_with_extension(directory, extension):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            files.append(os.path.join(directory, filename))
    return files


def get_selfie_paths(save_dir: Path | str = cfg.selfie_data.save_dir):
    selfies_dir = Path(save_dir)
    users_selfie_paths = []
    for user in selfies_dir.glob('*'):
        selfies = list_files_with_extension(user, '.jpg')
        users_selfie_paths.extend(selfies)
    return users_selfie_paths


def validate_selfie(img_path: Path | str):
    try:
        image = cv2.imread(img_path)
        if image is None:
            return img_path, False, "Unable to read the file as an image."
        else:
            return img_path, True, "File valid and can be read using cv2.imread()."
    except Exception as e:
        return img_path, False, f"Error: {str(e)}"
    

def validate_selfies(selfie_paths: list):
    validation_results = []
    with tqdm(desc="Validating selfies...", total=len(selfie_paths)) as pbar:
        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = [
                executor.submit(
                    validate_selfie,
                    path
                )
                for path in selfie_paths
            ]
            for future in concurrent.futures.as_completed(futures):
                validation_results.append(future.result())
                pbar.update(1)
    return validation_results


def get_landmarks(img_path: Path):
    try:
        landmarker = vision.FaceLandmarker.create_from_options(options)
        image = mp.Image.create_from_file(img_path)
        landmarks = landmarker.detect(image).face_landmarks[0]
        landmarks_coordinates = []
        for landmark in landmarks:
            x = landmark.x
            y = landmark.y
            z = landmark.z
            landmarks_coordinates.append((x,y,z))
        length = len(landmarks_coordinates)
        return (img_path, landmarks_coordinates, length)
    except Exception as e:
        logging.error(f"Error processing image at path {image_path}: {str(e)}")
        return None


def get_all_landmarks(selfie_paths: list) -> list:
    """
        parameters
        selfie_paths : list
            list of all selfie paths we want to get facial landmarks for.

        returns
        list
            a list of lists that contain for each selfie: file path, landmarks, number of landmarks
    """
    landmark_results = []
    with tqdm(desc="Calculating landmarks...", total=len(selfie_paths)) as pbar:
        with ProcessPoolExecutor(max_workers = 8, max_tasks_per_child=4500) as executor:
            futures = {executor.submit(get_landmarks, path) for path in selfie_paths}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        landmark_results.append(result)
                except Exception as e:
                    # Log or store information about the failed task
                    logging.error(f"Error processing image: {str(e)}")
                pbar.update(1)
    return landmark_results


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print('Getting selfie paths...')
    selfiepaths = get_selfie_paths(cfg.selfie_data.save_dir)
    selfie_validation = validate_selfies(selfiepaths)
    df_validation = pd.DataFrame(selfie_validation, columns=['selfie_path', 'valid', 'error'])

    good_selfies = df_validation[df_validation.valid].selfie_path.tolist()
    selfie_landmarks = get_all_landmarks(good_selfies)
    df_landmarks = pd.DataFrame(selfie_landmarks, columns=['selfie_path', 'landmarks', 'length'])
    
    df_final = pd.merge(df_validation, df_landmarks, on = 'selfie_path', how = 'left')
    df_final.to_csv('../../results/valid_selfies_w_landmark.csv')


if __name__ == "__main__":
    main()
    print("Done!")
