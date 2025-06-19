import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# === 設定區 ===
# 專案資料夾
projFolder = "training_workspace"
# 正樣本來源資料夾
imgFolder = os.path.join(projFolder, "positives")
# 增強後圖片輸出資料夾
outputFolder = os.path.join(projFolder, "aug_positives")
# 產生的圖片格式
imageKeepType = "jpg"
# 每張正樣本要產生幾張新圖片
numAugment = 6

# Augmentation 參數
aug_whitening = False
aug_rotation = 16
aug_w_shift = 0.1
aug_h_shift = 0.1
aug_shear = 0.1
aug_zoom = 0.05
aug_h_flip = True
aug_v_flip = False
aug_fillmode = "nearest"
# === 設定區結束 ===

def augImage(img_full_path, original_filename_no_ext):
    """
    對單一圖片進行資料增強，產生多張新圖片並儲存
    """
    try:
        img = load_img(img_full_path)
        x = img_to_array(img)
    except Exception as e:
        print(f"  [錯誤] 無法載入圖片 {img_full_path}：{e}，略過。")
        return

    datagen = ImageDataGenerator(
        zca_whitening=aug_whitening,
        rotation_range=aug_rotation,
        width_shift_range=aug_w_shift,
        height_shift_range=aug_h_shift,
        shear_range=aug_shear,
        zoom_range=aug_zoom,
        horizontal_flip=aug_h_flip,
        vertical_flip=aug_v_flip,
        fill_mode=aug_fillmode
    )

    x = x.reshape((1,) + x.shape)
    save_prefix = f"aug_{original_filename_no_ext}"
    i = 0
    for batch in datagen.flow(
        x, batch_size=1,
        save_to_dir=outputFolder,
        save_prefix=save_prefix,
        save_format=imageKeepType
    ):
        i += 1
        if i >= numAugment:
            break

if __name__ == "__main__":
    # 準備輸出資料夾
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
        print(f"已建立輸出資料夾: {outputFolder}")

    if not os.path.isdir(imgFolder):
        print(f"[錯誤] 找不到正樣本資料夾: {os.path.abspath(imgFolder)}")
        exit()

    print(f"開始進行資料增強，來源: {os.path.abspath(imgFolder)}")
    print(f"增強後圖片將儲存於: {os.path.abspath(outputFolder)}")

    processed_count = 0
    for file in os.listdir(imgFolder):
        filename_no_ext, file_extension = os.path.splitext(file)
        if file_extension.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            full_path_to_image = os.path.join(imgFolder, file)
            print(f"#{processed_count + 1} 處理: {file} ...")
            augImage(full_path_to_image, filename_no_ext)
            processed_count += 1

    print(f"\n資料增強完成，共處理 {processed_count} 張圖片。")
    print(f"所有增強圖片儲存於: {os.path.abspath(outputFolder)}")
