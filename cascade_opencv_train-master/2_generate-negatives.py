import time
import os

# === 設定區 ===
# 專案資料夾，所有產生的檔案或目錄皆會存於此
projFolder = "training_workspace"

# 是否將從已標註圖片產生的背景圖（neg_bg_from_annotated）加入 negatives.info
include_annotated_bg_in_negatives = True

# 基本負樣本來源資料夾（需包含不含目標物體的大圖）
base_negSourceFolders = [
    "dataset/pure_negatives",
]
# === 設定區結束 ===

# negatives.info 檔案路徑
negative_info_file_path = os.path.join(projFolder, "negatives.info")

# 動態組合所有負樣本來源資料夾
negSourceFolders = list(base_negSourceFolders)
if include_annotated_bg_in_negatives:
    path_to_annotated_bg = os.path.join(projFolder, "neg_bg_from_annotated")
    if os.path.isdir(os.path.abspath(path_to_annotated_bg)):
        negSourceFolders.append(path_to_annotated_bg)
        print(f"INFO: 已加入標註圖產生的背景資料夾: {path_to_annotated_bg}")
    else:
        print(f"WARNING: 設定為加入標註背景，但找不到資料夾: {os.path.abspath(path_to_annotated_bg)}")
        print(f"         請先執行 '1_labels_to_pos_neg_imgs.py' 並確保 'generateNegativeSource = True'")
else:
    print(f"INFO: 未將標註背景加入 negatives.info。")

if __name__ == "__main__":
    start_time = time.time()
    listed_files_count = 0

    with open(negative_info_file_path, 'w') as f_neg_info:
        print(f"產生 {negative_info_file_path}，列出所有原始負樣本圖片路徑。")
        print(f"搜尋負樣本來源資料夾: {negSourceFolders}")

        for folder_path in negSourceFolders:
            abs_folder_path = os.path.abspath(folder_path)
            if not os.path.isdir(abs_folder_path):
                print(f"警告：找不到資料夾或不是資料夾: {abs_folder_path}")
                continue

            print(f"處理資料夾: {abs_folder_path}")
            for imageName in os.listdir(abs_folder_path):
                if imageName.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    full_image_path = os.path.join(abs_folder_path, imageName)
                    if os.path.isfile(full_image_path):
                        # 寫入絕對路徑（OpenCV 支援絕對路徑）
                        path_to_write = os.path.abspath(full_image_path).replace('\\', '/')
                        f_neg_info.write(path_to_write + '\n')
                        listed_files_count += 1

    end_time = time.time()
    print(f"\n--- 負樣本描述檔產生總結 ---")
    print(f"總共列出 {listed_files_count} 張原始負樣本圖片於 {negative_info_file_path}")
    print(f"處理時間: {end_time - start_time:.2f} 秒")
    print(f"negatives.info 內的路徑皆為絕對路徑。")