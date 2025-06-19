import os
import cv2
import numpy as np
from xml.dom import minidom
from os.path import basename

# === 設定區 ===
# 標註 XML 檔案資料夾
xml_input_folder = "dataset/to_be_annotated_xmls"
# 已標註圖片資料夾
annotated_img_input_folder = "dataset/to_be_annotated"
# 純正樣本圖片資料夾（無需標註，直接用來當正樣本）
pure_positives_input_folder = "dataset/pure_positives"
# 要提取的標籤名稱
labelName_to_extract = "car"
# 輸出資料夾
output_folder_base = "training_workspace"
# 正樣本輸出子資料夾
pos_output_path_name = "positives"
# 從標註圖產生的負樣本背景輸出子資料夾
neg_bg_output_path_name = "neg_bg_from_annotated"
# 輸出圖片格式
image_keep_type = "jpg"
# 正樣本與負樣本的目標尺寸 (寬, 高)
target_size = (60, 60)
# 是否產生負樣本背景圖
generateNegativeSource = True
# === 設定區結束 ===

def saveROI(
    xml_path, img_path_base_folder, pos_save_path, neg_save_path,
    labelGrep_func, generateNeg_func, image_keep_type_func, target_size_func,
    positives_info_file_handle
):
    """
    依據單一 XML 標註檔案，裁切正樣本 ROI 並儲存，並產生負樣本背景圖。
    將正樣本資訊寫入 positives.info。
    回傳該 XML 檔案產生的正樣本數量。
    """
    print(f"    處理 {xml_path}，目標標籤：'{labelGrep_func}'")
    pos_counter_for_xml = 0

    try:
        xmldoc = minidom.parse(xml_path)
        item = xmldoc.documentElement

        # 取得圖片檔名
        img_filename_nodes = item.getElementsByTagName('filename')
        if not img_filename_nodes or not img_filename_nodes[0].firstChild:
            print(f"    [錯誤] 缺少 <filename> 標籤或內容為空")
            return 0
        img_filename = img_filename_nodes[0].firstChild.data
        full_img_path = os.path.join(img_path_base_folder, img_filename)
        img = cv2.imread(full_img_path)
        if img is None:
            print(f"    [錯誤] 無法讀取圖片: {full_img_path}")
            return 0

        height, width = img.shape[:2]
        neg_mask = np.zeros((height, width), dtype=np.uint8)  # 用於標記所有正樣本 ROI

        objects = item.getElementsByTagName('object')
        for obj_idx, obj in enumerate(objects):
            # 取得標籤名稱
            obj_name_nodes = obj.getElementsByTagName('name')
            if not obj_name_nodes or not obj_name_nodes[0].firstChild:
                print(f"    [警告] 物件 {obj_idx} 缺少 <name> 標籤")
                continue
            obj_name = obj_name_nodes[0].firstChild.data

            # 比對標籤名稱
            if obj_name.strip() == labelGrep_func.strip():
                # 取得邊界框
                bndbox_nodes = obj.getElementsByTagName('bndbox')
                if not bndbox_nodes:
                    print(f"        [警告] 物件 '{obj_name.strip()}' 缺少 <bndbox>")
                    continue
                bndbox = bndbox_nodes[0]
                try:
                    xmin = int(bndbox.getElementsByTagName('xmin')[0].firstChild.data)
                    ymin = int(bndbox.getElementsByTagName('ymin')[0].firstChild.data)
                    xmax = int(bndbox.getElementsByTagName('xmax')[0].firstChild.data)
                    ymax = int(bndbox.getElementsByTagName('ymax')[0].firstChild.data)
                except Exception:
                    print(f"        [警告] 物件 '{obj_name.strip()}' 邊界框座標有誤")
                    continue

                # 邊界檢查
                if xmin >= xmax or ymin >= ymax:
                    print(f"        [警告] 邊界框座標無效: ({xmin},{ymin})-({xmax},{ymax})")
                    continue
                if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                    print(f"        [警告] 邊界框超出圖片範圍: ({xmin},{ymin})-({xmax},{ymax})，圖片尺寸: ({width},{height})")
                    continue

                # 裁切 ROI 並縮放
                roi = img[ymin:ymax, xmin:xmax]
                if roi.size == 0:
                    print(f"        [警告] ROI 為空，座標: ({xmin},{ymin})-({xmax},{ymax})")
                    continue
                resized_roi = cv2.resize(roi, target_size_func, interpolation=cv2.INTER_AREA)

                # 儲存正樣本圖片
                pos_filename = f"{basename(xml_path).replace('.xml', '')}_obj{obj_idx}_{obj_name.strip()}.{image_keep_type_func}"
                full_pos_save_path = os.path.join(pos_save_path, pos_filename)
                cv2.imwrite(full_pos_save_path, resized_roi)

                # 寫入 positives.info
                relative_pos_path = full_pos_save_path.replace('\\', '/')
                positives_info_file_handle.write(f"{relative_pos_path} 1 0 0 {target_size_func[0]} {target_size_func[1]}\n")
                pos_counter_for_xml += 1

                # 標記 ROI 於 neg_mask
                cv2.rectangle(neg_mask, (xmin, ymin), (xmax, ymax), (255), -1)

        # 產生負樣本背景圖
        if generateNeg_func and neg_save_path:
            if np.sum(neg_mask) > 0:
                neg_background_mask = cv2.bitwise_not(neg_mask)
                neg_bg_img_part = cv2.bitwise_and(img, img, mask=neg_background_mask)
                if cv2.countNonZero(cv2.cvtColor(neg_bg_img_part, cv2.COLOR_BGR2GRAY)) > 0:
                    neg_bg_filename = f"{basename(xml_path).replace('.xml', '')}_neg_bg.{image_keep_type_func}"
                    full_neg_bg_save_path = os.path.join(neg_save_path, neg_bg_filename)
                    cv2.imwrite(full_neg_bg_save_path, neg_bg_img_part)
                else:
                    print(f"    [資訊] 移除 ROI 後背景全黑，未產生背景圖。")
            else:
                print(f"    [資訊] 無標記 ROI，未產生背景圖。")
    except FileNotFoundError:
        print(f"    [錯誤] 找不到 XML 檔案: {xml_path}")
    except Exception as e:
        print(f"    [錯誤] 處理 XML 失敗: {e}")

    return pos_counter_for_xml

if __name__ == "__main__":
    # === 輸出資料夾準備 ===
    projFolder = output_folder_base
    pos_output_path = os.path.join(projFolder, pos_output_path_name)
    neg_bg_output_path = os.path.join(projFolder, neg_bg_output_path_name)
    os.makedirs(pos_output_path, exist_ok=True)
    if generateNegativeSource:
        os.makedirs(neg_bg_output_path, exist_ok=True)

    positives_info_file_path = "positives.info"  # 可改為 os.path.join(projFolder, "positives.info")

    total_pos_rois_generated = 0
    total_xml_files_processed = 0
    total_pure_positives_added = 0

    # === 處理 XML 標註檔案 ===
    with open(positives_info_file_path, 'w') as positives_info_file:
        if os.path.isdir(xml_input_folder) and os.path.isdir(annotated_img_input_folder):
            for file in os.listdir(xml_input_folder):
                if file.lower().endswith(".xml"):
                    xml_file_path = os.path.join(xml_input_folder, file)
                    print(f"\n處理 XML: {basename(xml_file_path)}")
                    labels_found_in_file = saveROI(
                        xml_file_path, annotated_img_input_folder,
                        pos_output_path, neg_bg_output_path,
                        labelName_to_extract, generateNegativeSource,
                        image_keep_type, target_size, positives_info_file
                    )
                    if labels_found_in_file and labels_found_in_file > 0:
                        total_pos_rois_generated += labels_found_in_file
                        total_xml_files_processed += 1
        else:
            print(f"[警告] 找不到 XML 或已標註圖片資料夾。")

        # === 處理純正樣本圖片 ===
        if os.path.isdir(pure_positives_input_folder):
            print(f"\n處理純正樣本資料夾: {pure_positives_input_folder}")
            for file in os.listdir(pure_positives_input_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    try:
                        img_path = os.path.join(pure_positives_input_folder, file)
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"    [警告] 無法讀取純正樣本圖片: {img_path}")
                            continue
                        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                        pure_pos_filename = f"pure_{file}"
                        full_pure_pos_save_path = os.path.join(pos_output_path, pure_pos_filename)
                        cv2.imwrite(full_pure_pos_save_path, resized_img)
                        relative_pure_pos_path = full_pure_pos_save_path.replace('\\', '/')
                        positives_info_file.write(f"{relative_pure_pos_path} 1 0 0 {target_size[0]} {target_size[1]}\n")
                        total_pure_positives_added += 1
                    except Exception as e:
                        print(f"    [錯誤] 處理純正樣本圖片 {file} 失敗: {e}")
        else:
            print(f"[警告] 找不到純正樣本資料夾。")

    # === 統計與總結 ===
    print(f"\n--- 執行總結 ---")
    print(f"處理到含有目標標籤('{labelName_to_extract}')的 XML 檔案數: {total_xml_files_processed}")
    print(f"由 XML 產生的正樣本數: {total_pos_rois_generated}")
    print(f"純正樣本數: {total_pure_positives_added}")
    print(f"positives.info 總筆數: {total_pos_rois_generated + total_pure_positives_added}")
    print(f"\n正樣本圖片儲存於: {os.path.abspath(output_folder_base)}")
    print(f"positives.info 路徑: {os.path.abspath(positives_info_file_path)}")
    if generateNegativeSource:
        print(f"負樣本背景圖儲存於: {os.path.abspath(neg_bg_output_path)}")
