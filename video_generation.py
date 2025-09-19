import os
import cv2

def generate_video_from_images(folder_path):
    """
    從資料夾中的圖片生成影片，並添加字幕。
    Args:
        folder_path (str): 包含圖片的資料夾路徑。
    """
    # 收集資料夾中的圖片檔案
    images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(".jpg")]
    images.sort()  # 確保圖片按照名稱排序

    if not images:
        print(f"資料夾 {folder_path} 中沒有圖片，跳過影片生成。")
        return

    # 讀取第一張圖片以獲取影片尺寸
    first_image = cv2.imread(images[0])
    height, width, layers = first_image.shape

    # 定義影片輸出名稱和編解碼器
    video_name = os.path.basename(folder_path) + ".mp4"
    video_path = os.path.join(folder_path, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 編碼器
    video = cv2.VideoWriter(video_path, fourcc, 15.0, (width, height))

    # 定義字幕文字（資料夾名稱）
    subtitle = os.path.basename(folder_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 白色文字
    outline_color = (0, 0, 0)  # 黑色邊線
    thickness = 2
    line_type = cv2.LINE_AA
    text_position = (50, 50)  # 左上角字幕位置
    outline_thickness = thickness + 2  # 邊線的厚度稍大

    # 將每張圖片寫入影片，並添加字幕
    for image_path in images:
        img = cv2.imread(image_path)
        if img is None:
            print(f"無法讀取圖片 {image_path}，跳過...")
            continue

        # 在圖片上添加黑色邊線（多次繪製以模擬邊線效果）
        cv2.putText(img, subtitle, text_position, font, font_scale, outline_color, outline_thickness, line_type)

        # 在黑色邊線上繪製白色文字
        cv2.putText(img, subtitle, text_position, font, font_scale, font_color, thickness, line_type)

        # 寫入影片
        video.write(img)

    video.release()
    print(f"已生成影片：{video_path}")