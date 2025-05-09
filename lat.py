import os
import time
import math
import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from imu_utils import capture_imu_data, save_data_to_csv

#########################################
# IMU Data Capture for User Reps
#########################################
def record_user_reps(folder, duration=5.04, port='COM4'):
    """
    Records five user motion repetitions and saves each as a CSV file.
    Returns a list of filenames and a corresponding list of colors for plotting.
    """
    user_rep_files = []
    user_colors = ['blue', 'red', 'green', 'pink', 'lightblue']
    for rep in range(5):
        print(f"\nPlease record your motion rep {rep+1}")
        time.sleep(0.5)
        print(f"Capturing IMU data for 5 seconds at 40Hz for User Rep {rep+1}...")
        imu_data = capture_imu_data(rep_id=rep+2, duration=duration, sample_rate=40, port=port, baudrate=115200)
        filename = os.path.join(folder, f"user_rep_{rep+1}.csv")
        save_data_to_csv(imu_data, filename)
        user_rep_files.append(filename)
    return user_rep_files, user_colors

#########################################
# Pitch Analysis
#########################################
def motion_analysis(ref_file, user_rep_files, user_colors):
    # Process reference rep
    df_ref = pd.read_csv(ref_file)
    time_ref = df_ref["timestamp"].values
    pitch_ref = df_ref["pitch"].values
    # Use pitch trough for alignment
    min_idx_ref = np.argmin(pitch_ref)
    trough_time_ref = time_ref[min_idx_ref]
    
    # Compute metrics for reference (based on pitch)
    rom_ref = np.max(pitch_ref) - np.min(pitch_ref)
    if len(time_ref) > 1:
        speeds_ref = np.diff(pitch_ref) / np.diff(time_ref)
        avg_speed_ref = np.mean(np.abs(speeds_ref))
    else:
        avg_speed_ref = 0
    print(f"Reference Rep: Range of Motion (Pitch) = {rom_ref:.2f}°, Average Speed = {avg_speed_ref:.2f}°/s")
    
    # Create subplots: 3 rows (Pitch, Roll, Yaw) x 2 columns (Original, Shifted)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # ----- PITCH PLOTS -----
    # Original pitch
    axes[0, 0].set_title("Original Pitch Angle Over Time (User Reps vs Reference Rep)")
    axes[0, 0].set_xlabel("Timestamp (s)")
    axes[0, 0].set_ylabel("Pitch Angle (°)")
    axes[0, 0].grid(True)
    axes[0, 0].plot(df_ref["timestamp"], df_ref["pitch"], label="Reference Rep", color='black')
    # Shifted pitch
    axes[0, 1].set_title("Time-Shifted Pitch (Aligned to Reference Rep)")
    axes[0, 1].set_xlabel("Timestamp (s)")
    axes[0, 1].set_ylabel("Pitch Angle (°)")
    axes[0, 1].grid(True)
    axes[0, 1].plot(df_ref["timestamp"], df_ref["pitch"], label="Reference Rep", color='black')
    
    # ----- ROLL PLOTS -----
    # Original roll
    axes[1, 0].set_title("Original Roll Angle Over Time (User Reps vs Reference Rep)")
    axes[1, 0].set_xlabel("Timestamp (s)")
    axes[1, 0].set_ylabel("Roll Angle (°)")
    axes[1, 0].grid(True)
    axes[1, 0].plot(df_ref["timestamp"], df_ref["roll"], label="Reference Rep", color='black')
    # Shifted roll
    axes[1, 1].set_title("Time-Shifted Roll (Aligned to Reference Rep)")
    axes[1, 1].set_xlabel("Timestamp (s)")
    axes[1, 1].set_ylabel("Roll Angle (°)")
    axes[1, 1].grid(True)
    axes[1, 1].plot(df_ref["timestamp"], df_ref["roll"], label="Reference Rep", color='black')
    
    # ----- YAW PLOTS -----
    # Original yaw
    axes[2, 0].set_title("Original Yaw Angle Over Time (User Reps vs Reference Rep)")
    axes[2, 0].set_xlabel("Timestamp (s)")
    axes[2, 0].set_ylabel("Yaw Angle (°)")
    axes[2, 0].grid(True)
    axes[2, 0].plot(df_ref["timestamp"], df_ref["yaw"], label="Reference Rep", color='black')
    # Shifted yaw
    axes[2, 1].set_title("Time-Shifted Yaw (Aligned to Reference Rep)")
    axes[2, 1].set_xlabel("Timestamp (s)")
    axes[2, 1].set_ylabel("Yaw Angle (°)")
    axes[2, 1].grid(True)
    axes[2, 1].plot(df_ref["timestamp"], df_ref["yaw"], label="Reference Rep", color='black')
    
    # Loop through user reps and add their data to each plot
    for i, user_file in enumerate(user_rep_files):
        df_user = pd.read_csv(user_file)
        time_user = df_user["timestamp"].values
        pitch_user = df_user["pitch"].values
        # Compute alignment using pitch trough
        min_idx_user = np.argmin(pitch_user)
        trough_time_user = time_user[min_idx_user]
        rom_user = np.max(pitch_user) - np.min(pitch_user)
        if len(time_user) > 1:
            speeds_user = np.diff(pitch_user) / np.diff(time_user)
            avg_speed_user = np.mean(np.abs(speeds_user))
        else:
            avg_speed_user = 0
        print(f"User Rep {i+1}: Range of Motion (Pitch) = {rom_user:.2f}°, Average Speed = {avg_speed_user:.2f}°/s")
        
        # Compute shifted timestamp based on alignment
        time_shift = trough_time_ref - trough_time_user
        df_user["shifted_timestamp"] = df_user["timestamp"] + time_shift
        
        # Plot pitch data
        axes[0, 0].plot(df_user["timestamp"], df_user["pitch"], linestyle='dashed',
                         color=user_colors[i], label=f"User Rep {i+1}")
        axes[0, 1].plot(df_user["shifted_timestamp"], df_user["pitch"], linestyle='dashed',
                         color=user_colors[i], label=f"Shifted User Rep {i+1}", alpha=0.7)
        
        # Plot roll data
        axes[1, 0].plot(df_user["timestamp"], df_user["roll"], linestyle='dashed',
                         color=user_colors[i], label=f"User Rep {i+1}")
        axes[1, 1].plot(df_user["shifted_timestamp"], df_user["roll"], linestyle='dashed',
                         color=user_colors[i], label=f"Shifted User Rep {i+1}", alpha=0.7)
        
        # Plot yaw data
        axes[2, 0].plot(df_user["timestamp"], df_user["yaw"], linestyle='dashed',
                         color=user_colors[i], label=f"User Rep {i+1}")
        axes[2, 1].plot(df_user["shifted_timestamp"], df_user["yaw"], linestyle='dashed',
                         color=user_colors[i], label=f"Shifted User Rep {i+1}", alpha=0.7)
    
    # Add legends to each subplot
    for row in range(3):
        axes[row, 0].legend()
        axes[row, 1].legend()
    
    plt.tight_layout()
    plt.show()

#########################################
# Quaternion and DTW Functions
#########################################
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def rotate_vector_by_quaternion(q, v):
    """
    Rotate vector v by quaternion q.
    """
    v_quat = np.array([0.0] + list(v))
    q_conj = quat_conjugate(q)
    rotated = quat_multiply(quat_multiply(q, v_quat), q_conj)
    return rotated[1:]

def rpy_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(roll), -math.sin(roll)],
                   [0, math.sin(roll), math.cos(roll)]])
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                   [0, 1, 0],
                   [-math.sin(pitch), 0, math.cos(pitch)]])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw), math.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def load_3d_path(csv_file, local_offset):
    df = pd.read_csv(csv_file)
    times = df["timestamp"].values
    coords_3d = []
    for _, row in df.iterrows():
        R = rpy_to_rotation_matrix(row["roll"], row["pitch"], row["yaw"])
        global_vec = R @ local_offset
        coords_3d.append(np.array(global_vec))
    # Shift coordinates so the starting point is at (0,0,0)
    initial_vec = coords_3d[0]
    shifted_coords = [tuple(vec - initial_vec) for vec in coords_3d]
    return times, shifted_coords

def segment_phases(times, coords_3d, pitch_values):
    idx_mid = np.argmin(pitch_values)
    times_up = times[:idx_mid+1]
    coords_up = coords_3d[:idx_mid+1]
    times_down = times[idx_mid:]
    coords_down = coords_3d[idx_mid:]
    return (times_up, coords_up), (times_down, coords_down), idx_mid

def l2_distance_3d(a, b):
    return math.dist(a, b)

def compute_dtw_distance_and_path(coords_ref, coords_user):
    distance, path = fastdtw(coords_ref, coords_user, dist=l2_distance_3d)
    return distance, path

def plot_user_dtw_alignment_3d(coords_ref, coords_user, dtw_path, ax,
                               title='', user_color='red', user_label='User Rep'):
    x_ref = [p[0] for p in coords_ref]
    y_ref = [p[1] for p in coords_ref]
    z_ref = [p[2] for p in coords_ref]
    x_user = [p[0] for p in coords_user]
    y_user = [p[1] for p in coords_user]
    z_user = [p[2] for p in coords_user]
    
    # Plot user rep as dashed line
    ax.plot(x_user, y_user, z_user, linestyle='dashed', color=user_color, label=user_label)
    
    # Plot matching lines (downsampled for clarity)
    for i, (i_ref, i_user) in enumerate(dtw_path[::5]):
        p_ref = coords_ref[i_ref]
        p_user = coords_user[i_user]
        ax.plot([p_ref[0], p_user[0]],
                [p_ref[1], p_user[1]],
                [p_ref[2], p_user[2]],
                color='gray', linestyle='--', alpha=0.5)
    
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

#########################################
# Position Feedback Functions
#########################################
def compute_positions(file_path):
    """
    Computes positions from acceleration data using numerical integration.
    """
    df = pd.read_csv(file_path)
    n = len(df)
    time_arr = df['timestamp'].values
    velocities = np.zeros((n, 3))
    positions = np.zeros((n, 3))
    
    for i in range(1, n):
        dt = time_arr[i] - time_arr[i-1]
        if dt <= 0:
            dt = 1e-3
        q = np.array([df.loc[i, "qw"], df.loc[i, "qx"], df.loc[i, "qy"], df.loc[i, "qz"]])
        q = q / np.linalg.norm(q)
        local_acc = np.array([df.loc[i, "ax"], df.loc[i, "ay"], df.loc[i, "az"]])
        global_acc = rotate_vector_by_quaternion(q, local_acc)
        velocities[i] = velocities[i-1] + global_acc * dt
        positions[i] = positions[i-1] + velocities[i-1] * dt + 0.5 * global_acc * (dt ** 2)
    
    return time_arr, positions

def compare_motions(ref_positions, user_positions, threshold=0.2):
    diff = np.abs(ref_positions - user_positions)
    max_diff = np.max(diff, axis=0)
    prompts = []
    
    if max_diff[1] > 0.05:
        idy = np.argmax(np.abs(ref_positions[:, 1] - user_positions[:, 1]))
        if ref_positions[idy, 0] > user_positions[idy, 0]:
            prompts.append("Your arm is too far back, perform the motion side on.")
        else:
            prompts.append("Your arm is going forward too much, perform the motion side on.")

    
    if max_diff[0] > 0.15:
        prompts.append("Please perform the full range of motion without exceeding nor fall short of range.")
    
    if max_diff[2] > threshold:
        prompts.append("You are not completing the full hammer curl motion. Please perform the complete motion.")
    
    if not prompts:
        prompts.append("Good job, your motion is extremely accurate!")
    
    return prompts

#########################################
# Angular Speed and Stability Functions
#########################################
def compute_angular_speed(file_path):
    df = pd.read_csv(file_path)
    timestamps = df['timestamp'].values
    n = len(df)
    ang_speeds = []
    mid_times = []
    
    for i in range(1, n):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0:
            dt = 1e-3
        q1 = np.array([df.loc[i-1, "qw"], df.loc[i-1, "qx"], df.loc[i-1, "qy"], df.loc[i-1, "qz"]])
        q2 = np.array([df.loc[i, "qw"], df.loc[i, "qx"], df.loc[i, "qy"], df.loc[i, "qz"]])
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        q_rel = quat_multiply(q2, quat_conjugate(q1))
        q_rel = q_rel / np.linalg.norm(q_rel)
        q0 = np.clip(q_rel[0], -1.0, 1.0)
        angle = 2 * np.arccos(q0)
        angular_speed = angle / dt
        ang_speeds.append(angular_speed)
        mid_times.append((timestamps[i] + timestamps[i-1]) / 2)
    
    return np.array(mid_times), np.array(ang_speeds)

def compare_peak_speed(ref_ang_speeds, user_ang_speeds_list, tolerance=0.5):
    ref_peak = np.max(ref_ang_speeds)
    messages = []
    for i, user_ang_speeds in enumerate(user_ang_speeds_list):
        user_peak = np.max(user_ang_speeds)
        print(f"\nReference Peak Angular Speed: {ref_peak:.2f} rad/s")
        print(f"User Rep {i+1} Peak Angular Speed: {user_peak:.2f} rad/s")
        if user_peak > ref_peak + tolerance:
            messages.append(f"User Rep {i+1}: Your angular speed is too high; please control your speed!")
        elif user_peak < ref_peak - tolerance:
            messages.append(f"User Rep {i+1}: Your angular speed is too low; try to increase your speed.")
        else:
            messages.append(f"User Rep {i+1}: Good speed! Your motion is well controlled.")
    return messages

def assess_stability(ang_speeds):
    mean_speed = np.mean(ang_speeds)
    std_speed = np.std(ang_speeds)
    cv = std_speed / mean_speed if mean_speed != 0 else 0
    return mean_speed, std_speed, cv

def compare_stability(ref_ang_speeds, user_ang_speeds, tolerance_cv=0.2):
    ref_mean, ref_std, ref_cv = assess_stability(ref_ang_speeds)
    user_mean, user_std, user_cv = assess_stability(user_ang_speeds)
    print(f"Reference Angular Speed: Mean = {ref_mean:.2f}, Std = {ref_std:.2f}, CV = {ref_cv:.2f}")
    print(f"User Angular Speed: Mean = {user_mean:.2f}, Std = {user_std:.2f}, CV = {user_cv:.2f}")
    if user_cv > ref_cv + tolerance_cv:
        return "Your motion appears unstable and too fast. Try to maintain a smoother, more controlled speed."
    elif user_cv < ref_cv - tolerance_cv:
        return "Your motion is too slow. Try to maintain a faster speed."
    else:
        return "Good stability! Your motion is consistent."

#########################################
# DTW Analysis Function
#########################################
def dtw_analysis(ref_file, user_rep_files, user_colors, local_offset=np.array([0.0, 0.0, 0.3])):
    df_ref = pd.read_csv(ref_file)
    time_ref, coords_ref = load_3d_path(ref_file, local_offset)
    pitch_ref = df_ref["pitch"].values
    (ref_up, ref_down, idx_ref_mid) = segment_phases(time_ref, coords_ref, pitch_ref)
    coords_ref_up = ref_up[1]
    coords_ref_down = ref_down[1]
    
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    ax1.plot([p[0] for p in coords_ref_up], [p[1] for p in coords_ref_up], [p[2] for p in coords_ref_up],
             color='black', label='Reference Rep')
    ax2.plot([p[0] for p in coords_ref_down], [p[1] for p in coords_ref_down], [p[2] for p in coords_ref_down],
             color='black', label='Reference Rep')
    ax3.plot([p[0] for p in coords_ref], [p[1] for p in coords_ref], [p[2] for p in coords_ref],
             color='black', label='Reference Rep')
    
    for i, user_file in enumerate(user_rep_files):
        df_user = pd.read_csv(user_file)
        time_user, coords_user = load_3d_path(user_file, local_offset)
        pitch_user = df_user["pitch"].values
        (user_up, user_down, _) = segment_phases(time_user, coords_user, pitch_user)
        
        # Up phase DTW
        dist_up, path_up = compute_dtw_distance_and_path(coords_ref_up, user_up[1])
        min_local_distance_up = min(l2_distance_3d(coords_ref_up[i_ref], user_up[1][i_user])
                                    for i_ref, i_user in path_up)
        adjusted_dist_up = dist_up - min_local_distance_up
        
        # Down phase DTW
        dist_down, path_down = compute_dtw_distance_and_path(coords_ref_down, user_down[1])
        min_local_distance_down = min(l2_distance_3d(coords_ref_down[i_ref], user_down[1][i_user])
                                      for i_ref, i_user in path_down)
        adjusted_dist_down = dist_down - min_local_distance_down
        avg_adjusted_dist = (adjusted_dist_up + adjusted_dist_down) / 2.0

        print(f"\nUser Rep {i+1} DTW Analysis:")
        print(f"Original DTW distance (Up phase):   {dist_up:.3f}")
        print(f"Original DTW distance (Down phase): {dist_down:.3f}")
        print(f"Adjusted DTW distance (Up phase):   {adjusted_dist_up:.3f}")
        print(f"Adjusted DTW distance (Down phase): {adjusted_dist_down:.3f}")
        print(f"Average Adjusted DTW distance:      {avg_adjusted_dist:.3f}")
        if adjusted_dist_up < 4:
            print("Excellent! Your upwards motion closely matches the reference.")
        else:
            print("Noticeable deviation in upwards motion. Try to align your form more closely.")
        if adjusted_dist_down < 9:
            print("Excellent! Your downwards motion closely matches the reference.")
        else:
            print("Noticeable deviation in downwards motion. Try to align your form more closely.")
        if avg_adjusted_dist < 6:
            print("Good job! Your overall motion is closely matches the reference.")
        else:
            print("There are areas for improvement in your motion as your overall motion does not closely match the reference.")

        plot_user_dtw_alignment_3d(coords_ref_up, user_up[1], path_up, ax1,
                                   title='Up Phase DTW Alignment',
                                   user_color=user_colors[i],
                                   user_label=f'User Rep {i+1}')
        plot_user_dtw_alignment_3d(coords_ref_down, user_down[1], path_down, ax2,
                                   title='Down Phase DTW Alignment',
                                   user_color=user_colors[i],
                                   user_label=f'User Rep {i+1}')
        ax3.plot([p[0] for p in coords_user], [p[1] for p in coords_user], [p[2] for p in coords_user],
                 label=f'User Rep {i+1}', linestyle='dashed', color=user_colors[i])
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax3.set_title('Full Motion 3D Trajectory (Reference Rep vs User Reps)')
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")
    plt.tight_layout()
    plt.show()

#########################################
# Angular Speed and Stability Analysis
#########################################
def angular_speed_analysis(ref_file, user_rep_files, user_colors):
    # Compute angular speeds for the reference rep
    time_ref_ang, ref_ang_speeds = compute_angular_speed(ref_file)
    plt.figure(figsize=(10, 6))
    plt.plot(time_ref_ang, ref_ang_speeds, label="Reference Rep Angular Speed",
             color='black', marker='o', markersize=3)
    
    # Compute angular speeds for each user rep and collect them in a list
    user_ang_speeds_list = []
    for i, user_file in enumerate(user_rep_files):
        t_ang, user_ang = compute_angular_speed(user_file)
        plt.plot(t_ang, user_ang, linestyle='dashed', marker='o', markersize=3,
                 label=f"User Rep {i+1} Angular Speed", color=user_colors[i])
        user_ang_speeds_list.append(user_ang)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Speed (rad/s)")
    plt.title("Angular Speed vs Time (Reference Rep vs User Reps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Get feedback for each rep's peak angular speed
    feedback_messages = compare_peak_speed(ref_ang_speeds, user_ang_speeds_list, tolerance=0.5)
    for msg in feedback_messages:
        print(msg)

def stability_analysis(ref_file, user_rep_files):
    for i, user_file in enumerate(user_rep_files):
        _, user_ang_speeds = compute_angular_speed(user_file)
        feedback = compare_stability(compute_angular_speed(ref_file)[1], user_ang_speeds, tolerance_cv=0.2)
        print(f"Stability Analysis for User Rep {i+1}: {feedback}")

#########################################
# Main Function for Bicep Analysis
#########################################
def main(folder="motion_files", duration=5, port='COM4'):
    """
    Runs the full hammer curl analysis workflow:
    - Records five user reps,
    - Performs pitch analysis,
    - Computes DTW alignment,
    - Evaluates position feedback,
    - Analyzes angular speed and stability.
    """
    ref_file = os.path.join(folder, "reference.csv")
    user_rep_files, user_colors = record_user_reps(folder, duration, port)
    motion_analysis(ref_file, user_rep_files, user_colors)
    dtw_analysis(ref_file, user_rep_files, user_colors)
    
    print("\n----- XYZ Position Feedback Analysis -----")
    time_ref_pos, ref_positions = compute_positions(ref_file)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time_ref_pos, ref_positions[:, 0], label="Reference Rep", color='black')
    for i, user_file in enumerate(user_rep_files):
        time_user_pos, user_positions = compute_positions(user_file)
        feedback_messages = compare_motions(ref_positions, user_positions, threshold=0.2)
        print(f"\nXYZ Position Feedback for User Rep {i+1}:")
        for msg in feedback_messages:
            print(msg)
        axes[0].plot(time_user_pos, user_positions[:, 0],
                     linestyle='dashed', label=f"User Rep {i+1}", color=user_colors[i])
    axes[0].set_ylabel("X Position (m)")
    axes[0].set_title("X Position vs Time")
    axes[0].legend()
    
    axes[1].plot(time_ref_pos, ref_positions[:, 1], label="Reference Rep", color='black')
    for i, user_file in enumerate(user_rep_files):
        time_user_pos, user_positions = compute_positions(user_file)
        axes[1].plot(time_user_pos, user_positions[:, 1],
                     linestyle='dashed', label=f"User Rep {i+1}", color=user_colors[i])
    axes[1].set_ylabel("Y Position (m)")
    axes[1].set_title("Y Position vs Time")
    axes[1].legend()
    
    axes[2].plot(time_ref_pos, ref_positions[:, 2], label="Reference Rep", color='black')
    for i, user_file in enumerate(user_rep_files):
        time_user_pos, user_positions = compute_positions(user_file)
        axes[2].plot(time_user_pos, user_positions[:, 2],
                     linestyle='dashed', label=f"User Rep {i+1}", color=user_colors[i])
    axes[2].set_ylabel("Z Position (m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Z Position vs Time")
    axes[2].legend()
    plt.tight_layout()
    plt.show()
    
    angular_speed_analysis(ref_file, user_rep_files, user_colors)
    print("\n----- Stability and Control Analysis -----")
    stability_analysis(ref_file, user_rep_files)

if __name__ == '__main__':
    main()
