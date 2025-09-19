import os
import cv2
import json
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.arcline_path_utils import discretize_lane
from scipy.interpolate import interp1d

def find_image_coordinates(nusc, ordered_tokens, output_json_path, split='trainval'):
    # NuScenes 資料夾位置
    DATASET_DIR = "/data/NuScene/v1.0-"+split+"_meta"
    IMAGE_DIR = "/data/NuScene/v1.0-" + split

    samples_sweeps_data_path = "/data/NuScene/v1.0-" + split+"_meta/v1.0-" + split +"/sample_data.json"
    # 讀取 samples_sweeps_data.json
    with open(samples_sweeps_data_path, 'r') as f:
        samples_sweeps_data = json.load(f)

    # 讀取 ego_pose.json
    metaData = '/data/NuScene/v1.0-' + split + '_meta/v1.0-' + split +'/'
    with open(metaData + 'ego_pose.json') as f:
        ego_pose_data = json.load(f)

    output = []

    sample_sweep_info = next((item for item in samples_sweeps_data if item['token'] == ordered_tokens[0]), None)
    if sample_sweep_info:
        scene_sample_token = sample_sweep_info['sample_token']
    scene_sample = nusc.get('sample', scene_sample_token)
    # 取得 scene_token
    scene_token = scene_sample['scene_token']

    # 用 scene_token 取得 scene，再從 scene 拿到 log_token
    scene_record = nusc.get('scene', scene_token)
    log_token = scene_record['log_token']

    # 取得 log，從中讀取地圖名稱
    log_record = nusc.get('log', log_token)
    map_name = log_record['location']
    print(map_name)
    nusc_map = NuScenesMap(dataroot="/data/NuScene", map_name=map_name)

    for token in ordered_tokens:
        # 找到 sample_token
        sample_sweep_info = next((item for item in samples_sweeps_data if item['token'] == token), None)
        if sample_sweep_info is None:
            continue
        sample_token = sample_sweep_info['sample_token']
        sample = nusc.get('sample', sample_token)

        # 找到 ego_pose
        ego_pose  = next((item for item in ego_pose_data if item['token'] == token), None)
        if ego_pose is None:
            continue
        

        # 取得 Ego Pose 俯仰角（Pitch）與 滾轉角（Roll）
        rotation = Quaternion(ego_pose['rotation'])
        pitch = np.arcsin(2 * (rotation.w * rotation.y - rotation.z * rotation.x))  # Pitch
        roll = np.arcsin(2 * (rotation.w * rotation.x + rotation.y * rotation.z))  # Roll
        pitch_degrees = np.degrees(pitch)
        roll_degrees = np.degrees(roll)
        #print(f"Ego Pitch Angle: {pitch_degrees:.2f} degrees")
        #print(f"Ego Roll Angle: {roll_degrees:.2f} degrees")


        # 取得前視相機資料
        camera_token = sample['data']['CAM_FRONT']
        camera_data = nusc.get('sample_data', camera_token)

        # 取得相機內外參數
        calibrated_sensor = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
        camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])

        # 相機外參矩陣
        camera_extrinsic = transform_matrix(
            calibrated_sensor['translation'],
            Quaternion(calibrated_sensor['rotation']),
            inverse=True
        )

        # 取得 Ego Pose 轉換矩陣
        ego_to_world = transform_matrix(
            ego_pose['translation'],
            Quaternion(ego_pose['rotation']),
            inverse=True
        )


        # 載入相機影像
        image_path = f"{IMAGE_DIR}/{camera_data['filename']}"
        image = cv2.imread(image_path)

        # 取得車道線
        x, y = ego_pose['translation'][:2]
        radius = 50.0  # 搜尋半徑

        lanes = nusc_map.get_records_in_radius(x, y, radius, layer_names=['lane'])

        total_lane_points = []

        # 調整後的車道線點
        for lane_token in lanes['lane']:
            lane_record = nusc_map.get('lane', lane_token)
            #print(lane_record)

            # 同時處理左 / 右 divider
            divider_types = ['left_lane_divider_segment_nodes', 'right_lane_divider_segment_nodes']
            for divider_type in divider_types:
                divider_nodes = lane_record[divider_type]
                if len(divider_nodes) < 2:
                    continue

                lane_points_world = []
                ego_lane_points = []  # 存 Ego 坐標
                camera_lane_points = []  # 存 Camera 坐標
                image_points = []

                # 取出 X, Y, (Z 如無則用0)
                x_coords = np.array([node['x'] for node in divider_nodes])
                y_coords = np.array([node['y'] for node in divider_nodes])
                z_coords = np.zeros_like(x_coords)  # 如果 map 有 z，可以改成 [node['z'] for node in divider_nodes]

                # 累積距離
                distances = np.cumsum(
                    np.sqrt(
                        np.diff(x_coords, prepend=x_coords[0])**2 +
                        np.diff(y_coords, prepend=y_coords[0])**2 +
                        np.diff(z_coords, prepend=z_coords[0])**2
                    )
                )

                # 每 4 m 取一個插值點
                new_distances = np.arange(distances[0], distances[-1], 4)

                # 線性插值
                interp_x = interp1d(distances, x_coords, kind='linear')
                interp_y = interp1d(distances, y_coords, kind='linear')
                interp_z = interp1d(distances, z_coords, kind='linear')

                new_x = interp_x(new_distances)
                new_y = interp_y(new_distances)
                new_z = interp_z(new_distances)

                for xi, yi, zi in zip(new_x, new_y, new_z):
                    lane_points_world.append(np.array([xi, yi, zi, 1]))

                #print(f"Total lane points collected: {len(lane_points_world)}")

                #print(np.tan(pitch))

                for point in lane_points_world:
                    # **World → Ego 坐標轉換**
                    world_point = np.array([point[0], point[1], point[2], 1])
                    ego_point = np.dot(ego_to_world, world_point)  # 轉換為 Ego 坐標
                    ego_lane_points.append(ego_point[:3])  # 只保留 X, Y, Z
                    '''
                    # **在 Ego 坐標內應用 Pitch & Roll 修正**
                    ego_point[2] -= np.tan(pitch) * ego_point[0]  # Pitch 影響前後坡度
                    ego_point[2] += np.tan(roll) * ego_point[1]  # Roll 影響左右傾斜
                    '''
                    # **Ego → Camera 坐標轉換**
                    camera_point = np.dot(camera_extrinsic, np.append(ego_point[:3], 1))
                    camera_lane_points.append(camera_point[:3])  # 只保留 X, Y, Z

                    # **Camera → Image 平面（乘上相機內參）**
                    if camera_point[2] > 0:
                        pixel_coords = camera_intrinsic @ camera_point[:3]
                        pixel_coords /= pixel_coords[2]

                        x, y = int(pixel_coords[0]), int(pixel_coords[1])
                        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                            image_points.append((x, y))
                
                if len(image_points) > 0:
                    image_points_sorted = sorted(image_points, key=lambda point: point[1])
                    total_lane_points.append(image_points_sorted)
        
        # 儲存當前 token 的 image_coordinates
        output.append({
            "token": token,
            "lanes": total_lane_points
        })



    # 輸出 json
    with open(output_json_path, 'w') as out_file:
        json.dump(output, out_file, indent=2)


