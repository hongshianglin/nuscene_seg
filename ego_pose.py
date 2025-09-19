import math
import os
import json
import numpy as np
import requests
from pyquaternion import Quaternion

def calculate_displacement(pose1, pose2):
    return np.linalg.norm(np.array(pose2) - np.array(pose1))

def convert_trajectory_to_route(trajectory, split_length=1):
    route = [trajectory[0]]  # 第一筆 future pose
    trajectory_length = 0
    for pose_index in range(len(trajectory)-1):
        displacement = calculate_displacement(trajectory[pose_index], trajectory[pose_index + 1])
        trajectory_length += displacement

    split_count = int(trajectory_length / split_length) + 1
    for split_index in range(split_count-1):
        if len(trajectory) < 2:
            break
        remaining_split_length = split_length
        for pose_index in range(len(trajectory) - 1):
            displacement = calculate_displacement(trajectory[pose_index], trajectory[pose_index + 1])
            if displacement < remaining_split_length:
                remaining_split_length -= displacement
            else:
                ratio = remaining_split_length / displacement
                split_point = (
                    np.array(trajectory[pose_index]) * (1 - ratio) + 
                    np.array(trajectory[pose_index + 1]) * ratio
                ).tolist()
                route.append(split_point)
                trajectory = [split_point] + trajectory[pose_index + 1:]
                break
            if pose_index == len(trajectory) - 2: # Remaining split length cross the last segment, add the last point
                route.append(trajectory[pose_index+1])
                trajectory = []
    return route

def convert_enu_to_latlon(gps, filename, log_json_path):
    logfile = filename.split('/')[-1].split('__')[0]

    # loading log.json
    with open(log_json_path, 'r') as f:
        logs = json.load(f)

    location = None
    for log in logs:
        if log['logfile'] == logfile:
            location = log['location']
            break

    if location is None:
        raise ValueError(f"[ERROR] Logfile {logfile} not found in log.json")

    location_to_origin = {
        "boston-seaport": [42.336849169438615, -71.05785369873047],
        "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
        "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
        "singapore-queenstown": [1.2782562240223188, 103.76741409301758]
    }

    if location not in location_to_origin:
        raise ValueError(f"[ERROR] Unknown location: {location}")

    origin_lat, origin_lon = location_to_origin[location]
    east, north = gps

    lat = origin_lat + (north / 111111)
    lon = origin_lon + (east / (111111 * math.cos(math.radians(origin_lat))))

    return lat, lon

def convert_latlon_to_enu(lat, lon, filename, log_json_path):
    logfile = filename.split('/')[-1].split('__')[0]

    # loading log.json
    with open(log_json_path, 'r') as f:
        logs = json.load(f)

    location = None
    for log in logs:
        if log['logfile'] == logfile:
            location = log['location']
            break

    if location is None:
        raise ValueError(f"[ERROR] Logfile {logfile} not found in log.json")

    location_to_origin = {
        "boston-seaport": [42.336849169438615, -71.05785369873047],
        "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
        "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
        "singapore-queenstown": [1.2782562240223188, 103.76741409301758]
    }

    if location not in location_to_origin:
        raise ValueError(f"[ERROR] Unknown location: {location}")

    origin_lat, origin_lon = location_to_origin[location]

    north = (lat - origin_lat) * 111111
    east = (lon - origin_lon) * 111111 * math.cos(math.radians(origin_lat))

    return east, north


def crawl_osm_route(trajectory, filenames, log_json_path='/data/NuScene/v1.0-trainval_meta/v1.0-trainval/log.json'):
    start_latlon = convert_enu_to_latlon(trajectory[0], filenames[0], log_json_path)
    end_latlon = convert_enu_to_latlon(trajectory[-1], filenames[-1], log_json_path)

    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_latlon[1]},{start_latlon[0]};{end_latlon[1]},{end_latlon[0]}?overview=full&geometries=geojson"
    response = requests.get(osrm_url)
    data = response.json()

    route_osm = []
    if "routes" in data and len(data["routes"]) > 0:
        coordinates = data["routes"][0]["geometry"]["coordinates"]
        enu_coordinates = []
        for coord in coordinates:
            lon, lat = coord
            try:
                east, north = convert_latlon_to_enu(lat, lon, filenames[0], log_json_path)
                enu_coordinates.append([east, north])
            except Exception as e:
                print(f"[ERROR] Failed to convert {lat}, {lon}: {e}")
                enu_coordinates.append([None, None])

        route_osm = enu_coordinates
    return route_osm

def process_ego_pose_data(ordered_tokens, folder_path, output_file_name, filenames, sample_data, split='trainval'):
    """
    處理 ego_pose 數據，計算速度和角速度，並輸出至指定 JSON。
    第一張影像的 speed 和 yaw_rate 設為 0，其後每張皆由當前 vs 前一張差異計算。
    Args:
        ordered_tokens (list): 已排序的 token 列表。
        folder_path (str): 資料夾路徑。
        output_file_name (str): 輸出 JSON 檔案名稱。
        filenames (list): 檔案名稱列表。
        sample_data (list): sample_data.json 的資料。
    """
    # 讀取 ego_pose.json
    metaData = '/data/NuScene/v1.0-' + split +'_meta/v1.0-' + split + '/'
    with open(metaData + 'ego_pose.json') as f:
        ego_poses = json.load(f)
    ego_pose_map = {pose['token']: pose for pose in ego_poses}
    ordered_ego_poses = [ego_pose_map[token] for token in ordered_tokens]

    # 若最後一筆沒有對應檔案 (unknown)，則刪除
    if filenames and filenames[-1] == 'unknown':
        ordered_ego_poses = ordered_ego_poses[:-1]
        filenames = filenames[:-1]

    # 初始化
    trajectory = []
    frame_data = []

    # 讀 calibrated_sensor.json
    with open(metaData + 'calibrated_sensor.json') as f:
        calibrated_sensor = json.load(f)
    sample_data_map = {pose['token']: pose for pose in sample_data}
    sensor_map = {pose['token']: pose for pose in calibrated_sensor}
    sensor_token = sample_data_map[ordered_ego_poses[0]['token']]
    scene_sensor = sensor_map[sensor_token["calibrated_sensor_token"]]

    num_poses = len(ordered_ego_poses)
    route = []
    route_osm = []
    log_json_path='/data/NuScene/v1.0-'+split+'_meta/v1.0-'+split+'/log.json'
    if num_poses > 0:
        # 第一張影像，速度與角速度皆為 0
        first = ordered_ego_poses[0]
        gps0 = (first['translation'][0], first['translation'][1])
        yaw = quaternion_to_heading(first['rotation'])
        frame_data.append({
            "delta_t": 0,
            "speed": 0,
            "yaw_rate": 0,
            "gps_gt": gps0,
            "heading_gt": yaw,
            "token": first['token'],
            "filename": filenames[0]
        })
        trajectory.append(gps0)

        # 其後每張以當前與前一張差值計算
        for i in range(1, num_poses):
            prev = ordered_ego_poses[i-1]
            curr = ordered_ego_poses[i]
            delta_t = (curr['timestamp'] - prev['timestamp']) / 1e6
            gps = (curr['translation'][0], curr['translation'][1])
            # --- 先計算原始值 ---
            raw_speed = (math.sqrt(
                (curr['translation'][0] - prev['translation'][0])**2 +
                (curr['translation'][1] - prev['translation'][1])**2 +
                (curr['translation'][2] - prev['translation'][2])**2
            ) / delta_t) if delta_t > 0 else 0
            raw_yaw  = (calculate_angular_velocity(prev['rotation'], curr['rotation'], delta_t)
                        if delta_t > 0 else 0)

            yaw = quaternion_to_heading(curr['rotation'])

            # --- speed 加上 0.1（只有 raw_speed ≠ 0 時）---
            if raw_speed != 0:
                speed = raw_speed + 0.1
            else:
                speed = raw_speed

            # --- yaw_rate 全部加上 0.01 ---
            yaw_rate = raw_yaw + 0.01

            frame_data.append({
                "delta_t": delta_t,
                "speed": speed,
                "yaw_rate": yaw_rate,
                "gps_gt": gps,
                "heading_gt": yaw,
                "token": curr['token'],
                "filename": filenames[i]
            })
            trajectory.append(gps)

        route = convert_trajectory_to_route(trajectory, split_length=2)
        route_osm = crawl_osm_route(trajectory, filenames, log_json_path)

    # 計算 rotation 和 translation
    cam_rot = [[0,0,0],[0,0,0],[0,0,0]]
    cam_trans = [0, 0, 0]
    nu2our = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    for i in range(3):
        t = 0
        for f in range(3):
            cam_rot[i][f] = float(nuscene_quat_to_matrix(scene_sensor['rotation'])[i][f])
            t += -cam_rot[i][f] * scene_sensor['translation'][f]
        cam_trans[i] = t
    cam_rot_arr = np.array(cam_rot)
    cam_rot_our = (cam_rot_arr @ np.linalg.inv(nu2our)).tolist()

    output = {
        "trajectory": trajectory,
        "route": route,
        "route_osm": route_osm,
        "ego2cam_rot": cam_rot_our,
        "ego2cam_trans": cam_trans,
        "calib_mat": scene_sensor['camera_intrinsic'],
        "frame_data": frame_data
    }

    # output JSON
    output_json_path = os.path.join(folder_path, output_file_name)
    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Ego pose 數據已輸出到 {output_json_path}")

def quaternion_multiply(q1, q2):
    """
    計算兩個四元數的乘積。
    Args:
        q1 (list): 第一個四元數。
        q2 (list): 第二個四元數。
    Returns:
        list: 四元數乘積。
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ]

def quaternion_conjugate(q):
    """
    計算四元數的共軛。
    Args:
        q (list): 四元數。
    Returns:
        list: 四元數的共軛。
    """
    w, x, y, z = q
    return [w, -x, -y, -z]

def nuscene_quat_to_matrix(rotation):
    """
    將 NuScenes 的 1x4 旋轉四元數轉換為 3x3 旋轉矩陣
    :param rotation: [w, x, y, z] 的四元數
    :return: 3x3 旋轉矩陣 (numpy array)
    """
    quat = Quaternion(rotation)  # 直接用四元數建立 Quaternion 物件
    rotation_matrix = quat.rotation_matrix  # 轉換為 3x3 旋轉矩陣
    nu2our = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    
    return np.linalg.inv(rotation_matrix)

def quaternion_to_heading(q):
    """
    Compute the vehicle heading (yaw) from a quaternion.
    Definition: x-axis points East, y-axis points North, z-axis points Up.
    Counter-clockwise is positive, range is [-pi, pi].

    Args:
        q (list or np.array): Quaternion [w, x, y, z]

    Returns:
        float: Heading (yaw) in radians
    """
    w, x, y, z = q
    
    # Extract yaw angle from quaternion
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)) 
    yaw = -(yaw - np.pi / 2) #NuScene: counter-clockwise, zero on east -> Ours: clockwise, zero on north
    
    return yaw    

def calculate_angular_velocity(q1, q2, delta_t):
    h1 = quaternion_to_heading(q1)
    h2 = quaternion_to_heading(q2)
    delta_theta = (h2 - h1 + np.pi) % (2 * np.pi) - np.pi # Ensure the case is correct: -180 and 180 -> should be zero 
    angular_velocity = delta_theta / delta_t
    
    return angular_velocity





