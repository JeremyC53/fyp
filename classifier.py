import os
import time
import pickle
import numpy as np
import pandas as pd
from imu_utils import capture_imu_data, save_data_to_csv

def extract_features(file_path):
    """
    Reads a CSV file and computes the mean and standard deviation for
    quaternion and acceleration columns. Returns a 14-element feature vector.
    """
    df = pd.read_csv(file_path)
    features = []
    for col in ['qw', 'qx', 'qy', 'qz', 'ax', 'ay', 'az']:
        features.append(df[col].mean())
        features.append(df[col].std())
    return np.array(features)

def record_reference_motion(folder, duration=5, port='COM4'):
    """
    Records a reference motion rep and saves it as 'reference.csv' in the given folder.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    print("Please record your reference motion")
    time.sleep(0.5)
    print(f"Capturing IMU data for 5 seconds at 40Hz for Reference Rep...")
    imu_data = capture_imu_data(rep_id=1, duration=duration, sample_rate=40, port=port, baudrate=115200)
    ref_filename = os.path.join(folder, "reference.csv")
    save_data_to_csv(imu_data, ref_filename)
    return ref_filename

def classify_motion(ref_filename, model_file='rf_motion_model.pkl'):
    """
    Loads a pre-trained Random Forest model, extracts features from the reference motion,
    and classifies the motion.
    """
    if not os.path.exists(ref_filename):
        print(f"Error: {ref_filename} does not exist.")
        return None

    features = extract_features(ref_filename).reshape(1, -1)
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    class_map = {0: 'bicep', 1: 'hammer', 2: 'lat'}
    predicted_class = class_map.get(prediction, "Unknown")
    confidence_percentage = probabilities[prediction] * 100
    
    print(f"Motion classified as: {predicted_class}")
    print(f"Confidence: {confidence_percentage:.2f}%")
    return predicted_class

def main():
    folder = "motion_files"
    duration = 5.04  # Duration in seconds
    ref_file = record_reference_motion(folder, duration)
    predicted_class = classify_motion(ref_file)
    if predicted_class and predicted_class.lower() == 'bicep':
        print("Motion corresponds to a bicep curl. Proceeding with bicep curl analysis...")
        # Import and run the bicep curl module's main function
        import bicep
        bicep.main(folder=folder, duration=duration)
    elif predicted_class and predicted_class.lower() == 'hammer':
        print("Motion corresponds to a hammer curl. Proceeding with hammer curl analysis...")
        # Import and run the hammer curl module's main function
        import hammer
        hammer.main(folder=folder, duration=duration)
    elif predicted_class and predicted_class.lower() == 'lat':
        print("Motion corresponds to a lateral raise. Proceeding with lateral raise analysis...")
        # Import and run the hammer curl module's main function
        import test
        test.main(folder=folder, duration=duration)
    else:
        print("Motion does not correspond to a bicep curl. Exiting.")

if __name__ == '__main__':
    main()
