import os
import json
import random

def add_gps_perturbation(json_path, max_offset=10.0):
    """
    讀取指定的 JSON 檔案，對 frame_data 裡面的 gps 座標做 ±max_offset (預設10) 的隨機擾動。
    （每一幀都基於上一幀的「原始」GPS 做擾動；第一幀也會對其完美 GPS 加上隨機偏移）
    擾動後的座標會新增到同一筆資料的 'gps_perturbed' 欄位。

    Args:
        json_path (str): 要讀取並覆寫的 JSON 檔案路徑。
        max_offset (float): 擾動的最大絕對值（單位同 GPS 座標單位）。
    """
    if not os.path.isfile(json_path):
        print(f"找不到 JSON 檔案: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'frame_data' not in data:
        print(f"此 JSON 不包含 frame_data 欄位: {json_path}")
        return

    prev_original = None  # 上一幀的「原始」GPS

    for frame in data['frame_data']:
        gps_original = frame.get('gps_gt')
        if not gps_original or len(gps_original) < 2:
            continue

        # 決定本幀的基準 GPS：若為第一幀，基準就是自己的完美 GPS；否則用上一幀的完美 GPS
        if prev_original is None:
            baseline = gps_original
        else:
            baseline = prev_original

        # 以 baseline 做隨機擾動
        offset_x = random.uniform(-max_offset, max_offset)
        offset_y = random.uniform(-max_offset, max_offset)
        gps_perturbed = [
            baseline[0] + offset_x,
            baseline[1] + offset_y,
        ]

        frame['gps_perturbed'] = gps_perturbed
        # 更新下一幀的 baseline 用原始 GPS
        prev_original = gps_original

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[完成] 已對 {json_path} 裡的 GPS 座標做擾動，結果存於 frame_data[].gps_perturbed")

