import os
from shutil import move
from find_token_and_segment import find_corresponding_entry, save_segment_metadata, segment_files_by_prev

def process_folder_recursive(nusc, folder_path, sample_data):
    """
    遞迴處理資料夾中的檔案與子資料夾，依據 prev 欄位進行分段。
    Args:
        folder_path (str): 要處理的資料夾路徑。
        sample_data (list): sample_data.json 的內容。
    """
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if os.path.isdir(item_path):
            # 遞迴處理子資料夾
            print(f"進入資料夾：{item_path}")
            process_folder_recursive(nusc, item_path, sample_data)
        else:
            # 忽略非目標檔案
            if not item_path.endswith(".jpg"):
                continue

            # 在處理多層資料夾時，針對目標資料夾進行分段
            segment_files_by_prev(nusc, folder_path, sample_data)
            break

def process_file_by_name(nusc, file_name, sample_data, source_folder):
    """
    處理單個檔案，直接輸出 Metadata。
    Args:
        file_name (str): 要處理的檔案名稱。
        sample_data (list): sample_data.json 的資料。
        source_folder (str): 檔案所在的資料夾路徑。
    """
    # 找到檔案對應的 sample_data 條目
    sample_entry = find_corresponding_entry(file_name, sample_data, by_token=False)
    if not sample_entry:
        print(f"檔案 {file_name} 無法在 sample_data.json 中找到對應資料，跳過。")
        return

    # 建立新資料夾
    folder_name = os.path.splitext(file_name)[0]
    folder_path = os.path.join(source_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # 移動檔案到新資料夾
    original_file_path = os.path.join(source_folder, file_name)
    new_file_path = os.path.join(folder_path, file_name)
    move(original_file_path, new_file_path)

    # 使用單個檔案的 token 輸出 Metadata，不生成影片
    output_file_name = f"{folder_name}.json"
    ordered_tokens = [sample_entry["token"]]
    filenames = [file_name]
    save_segment_metadata(nusc, folder_path, ordered_tokens, output_file_name, filenames, sample_data)

    print(f"已處理檔案 {file_name}，結果存放於 {folder_path}")
    
    
