import cv2
import os

# === 設定區 ===
# 專案資料夾
projFolder = "training_workspace"
# 增強後正樣本來源資料夾
aug_positives_input_folder = os.path.join(projFolder, "aug_positives")
# positives.info 檔案路徑
# 注意：此腳本會「覆寫」此檔案，只包含增強後的正樣本資訊
positiveDesc_file_path = "positives.info"
# === 設定區結束 ===

if __name__ == "__main__":
    # 專案根目錄（用於產生相對路徑）
    project_root = os.getcwd()

    # 檢查來源資料夾是否存在
    if not os.path.isdir(aug_positives_input_folder):
        print(f"[錯誤] 找不到增強正樣本資料夾: {os.path.abspath(aug_positives_input_folder)}")
        print("請先執行 3_augmentation.py。")
        exit()

    print(f"讀取增強正樣本來源: {os.path.abspath(aug_positives_input_folder)}")
    print(f"將「覆寫」並產生新的 positives.info: {os.path.abspath(positiveDesc_file_path)}")
  

    listed_count = 0
    # 使用 'w' 模式來覆寫 (write) positives.info 檔案
    with open(positiveDesc_file_path, 'w') as f_pos_info:
        for file in os.listdir(aug_positives_input_folder):
            filename, file_extension = os.path.splitext(file)
            if file_extension.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                source_aug_image_path = os.path.join(aug_positives_input_folder, file)

                try:
                    # 讀取圖片以取得尺寸
                    img = cv2.imread(source_aug_image_path)
                    if img is None:
                        print(f"  [警告] 無法讀取圖片: {source_aug_image_path}，略過。")
                        continue

                    img_height, img_width = img.shape[:2]
                    # 產生相對於專案根目錄的路徑，直接指向來源圖片
                    rel_path_aug = os.path.relpath(os.path.abspath(source_aug_image_path), start=project_root).replace('\\', '/')
                    info_line = f"{rel_path_aug} 1 0 0 {img_width} {img_height}\n"
                    f_pos_info.write(info_line)
                    listed_count += 1
                    if listed_count % 50 == 0:
                        print(f"  已處理 {listed_count} 張增強圖片...")

                except Exception as e:
                    print(f"  [錯誤] 處理 {source_aug_image_path} 失敗: {e}")

    print(f"\n處理完成，共將 {listed_count} 張增強圖片的路徑寫入 positives.info。")
    print(f"已產生新的 positives.info，其中只包含增強後的樣本資訊。")