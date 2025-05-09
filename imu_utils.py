import os
import time
import serial
import pandas as pd

def capture_imu_data(rep_id, duration=5, sample_rate=40, port='COM4', baudrate=115200):
    """
    Capture IMU data over a fixed duration and sample rate.
    Returns a list of dictionaries with IMU readings.
    """
    dt_fixed = 1.0 / sample_rate
    data = []
    try:
        ser = serial.Serial(port, baudrate, timeout=0.01)
    except Exception as e:
        print(f"Error opening serial port {port}: {e}")
        return data

    print(f"Starting IMU data capture for 5 seconds at {sample_rate} Hz.")
    max_samples = int(duration * sample_rate)
    sample_index = 0

    while sample_index < max_samples:
        try:
            line = ser.readline().decode('utf-8').strip()
        except Exception as e:
            print(f"Error reading from serial port: {e}")
            continue

        if not line:
            continue

        parts = line.split(',')
        if len(parts) != 10:
            parts = line.split()
        if len(parts) != 10:
            print(f"Unexpected data format: {parts}")
            continue

        try:
            qw, qx, qy, qz, roll, pitch, yaw, ax, ay, az = map(float, parts)
        except ValueError:
            print(f"Conversion error for: {parts}")
            continue

        timestamp = round(sample_index * dt_fixed, 4)
        record = {
            'rep_id': rep_id,
            'timestamp': round(timestamp, 3),
            'qw': round(qw, 2),
            'qx': round(qx, 2),
            'qy': round(qy, 2),
            'qz': round(qz, 2),
            'roll': round(roll, 2),
            'pitch': round(pitch, 2),
            'yaw': round(yaw, 2),
            'ax': round(ax, 2),
            'ay': round(ay, 2),
            'az': round(az, 2)
        }
        data.append(record)
        sample_index += 1

    ser.close()
    return data

def save_data_to_csv(data, filename):
    """
    Save a list of dictionaries to a CSV file.
    """
    if not data:
        print("No data to save.")
        return
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
