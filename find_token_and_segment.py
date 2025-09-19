import os
import json
from shutil import move
from video_generation import generate_video_from_images
from ego_pose import process_ego_pose_data
from lanes_coordinates import find_image_coordinates
from gps_perturbation import add_gps_perturbation

def find_corresponding_entry(identifier, sample_data, by_token=False):
    """
    根據檔名或 token 在 sample_data 中找到對應的條目。
    Args:
        identifier (str): 檔名或 token。
        sample_data (list): sample_data.json 的資料。
        by_token (bool): 是否根據 token 查找。
    Returns:
        dict or None: 找到的條目，或 None。
    """
    for entry in sample_data:
        if by_token:
            if entry["token"] == identifier:
                return entry
        else:
            if os.path.basename(entry["filename"]) == identifier:
                return entry
    return None

def save_segment_metadata(nusc, folder_path, ordered_tokens, output_file_name, filenames, sample_data, split):
    add_next_token_if_needed(ordered_tokens, filenames, sample_data)

    # Step 1: 先跑原本的 process_ego_pose_data，會輸出 base json
    output_json_path = os.path.join(folder_path, output_file_name)
    process_ego_pose_data(ordered_tokens, folder_path, output_file_name, filenames, sample_data, split)

    # Step 2: 跑 find_image_coordinates，生成 image_coordinates.json
    output_lanes_json = os.path.join(folder_path, "lanes.json")
    find_image_coordinates(
        nusc,
        ordered_tokens=ordered_tokens,
        output_json_path=output_lanes_json,
        split=split
    )

    # Step 3: merge image_coordinates 到主 json
    with open(output_json_path, 'r') as f:
        main_json = json.load(f)

    with open(output_lanes_json, 'r') as f:
        lane_json = json.load(f)

    # 轉成 dict 加速查找
    lane_dict = {item['token']: item['lanes'] for item in lane_json}

    # merge
    for frame in main_json['frame_data']:
        token = frame.get('token')
        frame['lanes'] = lane_dict.get(token, [])

    # 覆寫原本的 json
    with open(output_json_path, 'w') as f:
        json.dump(main_json, f, indent=2)
    if os.path.exists(output_lanes_json):
        os.remove(output_lanes_json)

    # Step 4: 若有多於一張圖片才生成影片
    if len(ordered_tokens) > 2:
        generate_video_from_images(folder_path)

    # 對剛完成的 JSON 做 GPS 擾動
    add_gps_perturbation(output_json_path, max_offset=10.0)

    """
    儲存 Metadata 並生成影片。
    Args:
        folder_path (str): 資料夾路徑。
        ordered_tokens (list): 已排序的 token 列表。
        output_file_name (str): 輸出 JSON 檔案名稱。
        filenames (list): 檔案名稱列表。
        sample_data (list): sample_data.json 的資料。
    """
    """
    add_next_token_if_needed(ordered_tokens, filenames, sample_data)
    process_ego_pose_data(ordered_tokens, folder_path, output_file_name, filenames)
    if len(ordered_tokens) > 2:  # 只有多於一張圖片時才生成影片
        generate_video_from_images(folder_path)
    """
        
def add_next_token_if_needed(ordered_tokens, filenames, sample_data):
    """
    檢查並添加最後一個 token 的 next token，同步更新 filenames。
    Args:
        ordered_tokens (list): 已排序的 token 列表。
        filenames (list): 檔案名稱列表。
        sample_data (list): sample_data.json 的資料。
    """
    if not ordered_tokens:
        return

    last_token = ordered_tokens[-1]
    last_entry = find_corresponding_entry(last_token, sample_data, by_token=True)
    if last_entry:
        if last_entry["next"]:
            ordered_tokens.append(last_entry["next"])
            filenames.append("unknown")  # 如果 next 沒有檔案名，用 "unknown" 填充
        else:
            print(f"Token {last_token} 沒有 next token，添加自己。")
            ordered_tokens.append(last_token)
            filenames.append(filenames[-1])  # 重複最後一個檔案名
