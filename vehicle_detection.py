import cv2
import numpy as np
import json
import os
import glob
import re
import math

class EnhancedParkingDetector:
    def __init__(self, cascade_path=None):
        # 統一從 model 資料夾載入車輛檢測器
        self.detectors = self.load_multiple_detectors(cascade_path)
        self.parking_zones = []
        self.total_spaces = 0
        self.zone_type = 'polygon' # 預設為多邊形
        
    def load_multiple_detectors(self, cascade_path):
        """載入多個 Haar 分類器以提升偵測率"""
        detectors = {}
        model_folder = 'model'
        possible_cascades = [os.path.join(model_folder, 'cascade5.xml')]
        if cascade_path:
            possible_cascades.insert(0, cascade_path)
        for cascade_file in possible_cascades:
            if os.path.exists(cascade_file):
                cascade = cv2.CascadeClassifier(cascade_file)
                if not cascade.empty():
                    detectors['car'] = cascade
                    print(f"已載入車輛檢測器: {cascade_file}")
                    break
        return detectors
    
    def preprocess_image(self, image):
        """影像前處理：灰階、直方圖均衡、模糊、對比增強"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(blurred)
    
    def detect_vehicles_multi_scale(self, image):
        """多組參數進行車輛偵測，合併重疊結果"""
        gray = self.preprocess_image(image)
        all_detections = []
        # --- detection_params 參數組，僅保留註解 ---
        detection_params = [
            # {'scaleFactor': 1.05, 'minNeighbors': 11, 'minSize': (100, 100), 'maxSize': (200, 200)}, #7
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (140, 140), 'maxSize': (600, 600)}, #5
            # {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (30, 30), 'maxSize': (100, 100)}, #1
        ]
        # --- 參數組註解保留 ---
        for params in detection_params:
            try:
                detections = self.detectors['car'].detectMultiScale(
                    gray,
                    scaleFactor=params['scaleFactor'],
                    minNeighbors=params['minNeighbors'],
                    minSize=params['minSize'],
                    maxSize=params['maxSize'],
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            except Exception:
                detections = []
            if len(detections) > 0:
                all_detections.extend(detections)
        if len(all_detections) == 0:
            return np.array([])
        # 合併重疊框
        return self.merge_overlapping_detections(np.array(all_detections), overlap_threshold=0.05)

    def merge_overlapping_detections(self, detections, overlap_threshold):
        """合併重疊偵測框"""
        if len(detections) == 0:
            return detections
        merged, used = [], [False] * len(detections)
        for i in range(len(detections)):
            if used[i]: continue
            group = [detections[i]]
            used[i] = True
            for j in range(i + 1, len(detections)):
                if used[j]: continue
                if self.calculate_iou(detections[i], detections[j]) > overlap_threshold:
                    group.append(detections[j])
                    used[j] = True
            merged.append(self.merge_detection_group(group))
        return np.array(merged)
    
    def calculate_iou(self, box1, box2):
        """計算兩個框的 IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x_left, y_top = max(x1, x2), max(y1, y2)
        x_right, y_bottom = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1, area2 = w1 * h1, w2 * h2
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0
    
    def merge_detection_group(self, group):
        """合併一組重疊框"""
        x = int(np.mean([det[0] for det in group]))
        y = int(np.mean([det[1] for det in group]))
        w = int(np.mean([det[2] for det in group]))
        h = int(np.mean([det[3] for det in group]))
        return [x, y, w, h]
    
    def detect_using_color_analysis(self, image, zone):
        """顏色分析法：非地面色比例超過 30% 判定有車"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # 僅處理多邊形
        if 'points' in zone:
            points = np.array(zone['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        else:
            return False # 如果沒有 points，無法處理

        zone_area = cv2.bitwise_and(image, image, mask=mask)
        hsv = cv2.cvtColor(zone_area, cv2.COLOR_BGR2HSV)
        lower_ground = np.array([0, 0, 50])
        upper_ground = np.array([180, 50, 200])
        ground_mask = cv2.inRange(hsv, lower_ground, upper_ground)
        non_ground_mask = cv2.bitwise_not(ground_mask)
        total_pixels = cv2.countNonZero(mask)
        non_ground_pixels = cv2.countNonZero(cv2.bitwise_and(non_ground_mask, mask))
        return (non_ground_pixels / total_pixels > 0.3) if total_pixels > 0 else False
    
    def detect_using_edge_analysis(self, image, zone):
        """邊緣分析法：邊緣密度超過 10% 判定有車"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # 僅處理多邊形
        if 'points' in zone:
            points = np.array(zone['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        else:
            return False # 如果沒有 points，無法處理

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        zone_edges = cv2.bitwise_and(edges, mask)
        edge_pixels = cv2.countNonZero(zone_edges)
        total_pixels = cv2.countNonZero(mask)
        return (edge_pixels / total_pixels > 0.1) if total_pixels > 0 else False
    
    def load_parking_zones(self, zones_file='parking_zones.json'):
        """載入停車格定義"""
        try:
            with open(zones_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.parking_zones = data['zones']
                self.total_spaces = len(self.parking_zones)
                self.zone_type = data.get('zone_type', 'polygon')
                print(f"已從 {os.path.basename(zones_file)} 載入 {self.total_spaces} 個停車格 (類型: {self.zone_type})")
                return True
        except FileNotFoundError:
            print(f"錯誤: 找不到停車格定義檔案: {zones_file}")
            return False
        except (json.JSONDecodeError, KeyError) as e:
            print(f"錯誤: 讀取或解析 JSON 檔案 '{zones_file}' 失敗: {e}")
            return False
    
    def point_in_polygon(self, point, polygon):
        """判斷點是否在多邊形內"""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def check_parking_occupancy_enhanced(self, image):
        """多方法融合判斷停車格佔用"""
        cars = self.detect_vehicles_multi_scale(image)
        occupied_spaces, occupied_zones, detection_details = 0, [], []
        for i, zone in enumerate(self.parking_zones):
            confidence_score, methods = 0, []
            if self.check_haar_detection(zone, cars):
                methods.append("Haar Detection")
                confidence_score += 0.2
            if self.detect_using_color_analysis(image, zone):
                methods.append("Color Analysis")
                confidence_score += 0.4
            if self.detect_using_edge_analysis(image, zone):
                methods.append("Edge Analysis")
                confidence_score += 0.4
            is_occupied = confidence_score >= 0.5
            if is_occupied:
                occupied_spaces += 1
                occupied_zones.append(i)
            detection_details.append({
                'zone_id': i,
                'is_occupied': is_occupied,
                'confidence': confidence_score,
                'methods': methods
            })
        return {
            'total_spaces': self.total_spaces,
            'occupied_spaces': occupied_spaces,
            'available_spaces': self.total_spaces - occupied_spaces,
            'occupied_zones': occupied_zones,
            'occupancy_rate': (occupied_spaces / self.total_spaces * 100) if self.total_spaces > 0 else 0,
            'detected_cars': len(cars),
            'detected_cars_rects': cars,
            'detection_details': detection_details
        }
    
    def check_haar_detection(self, zone, cars):
        """判斷 Haar 偵測結果是否與停車格重疊 (多邊形五點法)"""
        if len(cars) == 0:
            return False
        
        # 確保停車格是多邊形且有座標點
        if 'points' not in zone:
            return False
            
        zone_polygon = zone['points']
        for (cx, cy, cw, ch) in cars:
            # 取出車輛框的四個角點與中心點
            car_points = [
                (cx, cy), (cx + cw, cy), (cx, cy + ch), 
                (cx + cw, cy + ch), (cx + cw//2, cy + ch//2)
            ]
            # 檢查任一點是否在多邊形內
            for point in car_points:
                if self.point_in_polygon(point, zone_polygon):
                    return True
        return False
    
    def visualize_enhanced_parking(self, image, result):
        """繪製偵測結果與統計資訊"""
        output = image.copy()
        for i, zone in enumerate(self.parking_zones):
            detail = result['detection_details'][i]
            # 顏色與狀態
            if detail['is_occupied']:
                color = (0, 0, 255) if detail['confidence'] >= 0.7 else (0, 100, 255)
                status = f"OCCUPIED ({detail['confidence']:.1f})"
            else:
                color = (0, 255, 0)
                status = "AVAILABLE"
            
            # 畫停車格 (僅處理多邊形)
            if 'points' in zone:
                points = np.array(zone['points'], dtype=np.int32)
                cv2.polylines(output, [points], True, color, 4) # 線條加粗
                overlay = output.copy()
                cv2.fillPoly(overlay, [points], color)
                cv2.addWeighted(output, 0.8, overlay, 0.2, 0, output)
                center_x = int(np.mean([p[0] for p in zone['points']]))
                center_y = int(np.mean([p[1] for p in zone['points']]))
                # --- 大幅調整字體大小、粗細與位置 ---
                # 車位編號
                cv2.putText(output, f"#{zone.get('id', i+1)}", (center_x - 50, center_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
                # 狀態文字
                cv2.putText(output, status, (center_x - 150, center_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
                if detail['methods']:
                    methods_text = ", ".join(detail['methods'])
                    # 方法文字
                    cv2.putText(output, methods_text, (center_x - 160, center_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        # 畫車輛框
        for (x, y, w, h) in result['detected_cars_rects']:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 3)
        # 統計資訊
        info_text = [
            f"Total Spaces: {result['total_spaces']}",
            f"Occupied: {result['occupied_spaces']}",
            f"Available: {result['available_spaces']}",
            f"Occupancy Rate: {result['occupancy_rate']:.1f}%",
        ]
        overlay = output.copy()
        # --- 大幅調整統計資訊框大小 ---
        cv2.rectangle(overlay, (5, 5), (700, 280), (0, 0, 0), -1)
        cv2.addWeighted(output, 0.7, overlay, 0.3, 0, output)
        for i, text in enumerate(info_text):
            # --- 大幅調整統計資訊字體大小、粗細與位置 ---
            cv2.putText(output, text, (30, 70 + i * 55), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
        return output

def process_single_image():
    """處理單張指定的影像並顯示結果"""
    image_folder = 'parking_lot'
    json_folder = 'parking_lot_json'

    if not os.path.isdir(image_folder):
        print(f"錯誤：找不到影像資料夾 '{image_folder}'。")
        return

    image_files = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))

    if not image_files:
        print(f"在資料夾 '{image_folder}' 中找不到任何支援的影像檔案。")
        return

    print("找到以下影像檔案:")
    for i, file in enumerate(image_files):
        print(f"{i+1}. {os.path.basename(file)}")

    selected_image = None
    if len(image_files) == 1:
        selected_image = image_files[0]
        print(f"\n自動選擇: {os.path.basename(selected_image)}")
    else:
        try:
            choice = int(input("請選擇影像 (輸入數字): ")) - 1
            if 0 <= choice < len(image_files):
                selected_image = image_files[choice]
            else:
                print("選擇無效，使用第一個影像")
                selected_image = image_files[0]
        except (ValueError, IndexError):
            print("輸入無效，使用第一個影像")
            selected_image = image_files[0]

    image_basename = os.path.basename(selected_image)
    json_filename = os.path.splitext(image_basename)[0] + '.json'
    json_path = os.path.join(json_folder, json_filename)

    print(f"\n將處理影像: {selected_image}")
    print(f"將使用定義檔: {json_path}")

    detector = EnhancedParkingDetector()
    if not detector.load_parking_zones(json_path):
        print(f"\n錯誤：無法載入停車格定義。")
        print(f"請確認 '{json_path}' 存在且格式正確。")
        print(f"您可能需要先為影像 '{image_basename}' 執行 `class.py` 來產生定義檔。")
        return

    image = cv2.imread(selected_image)
    if image is None:
        print(f"無法載入影像: {selected_image}")
        return

    print(f"正在處理影像...")
    result = detector.check_parking_occupancy_enhanced(image)
    output = detector.visualize_enhanced_parking(image, result)

    cv2.namedWindow('Enhanced Parking Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Enhanced Parking Detection', 1400, 900)
    cv2.imshow('Enhanced Parking Detection', output)

    print("\n=== 檢測報告 ===")
    print(f"總停車格: {result['total_spaces']}")
    print(f"已佔用: {result['occupied_spaces']}")
    print(f"可用空位: {result['available_spaces']}")
    print(f"佔用率: {result['occupancy_rate']:.1f}%")
    print(f"檢測到車輛特徵: {result['detected_cars']}")
    print("\n=== 詳細分析 ===")
    for detail in result['detection_details']:
        if detail['is_occupied']:
            zone_info = detector.parking_zones[detail['zone_id']]
            zone_id_str = zone_info.get('id', 'N/A')
            print(f"停車格 #{detail['zone_id']+1} (ID: {zone_id_str}): 已佔用 (信心度: {detail['confidence']:.2f}) - {', '.join(detail['methods'])}")

    print("\n按任意鍵退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scan_all_for_vacancies():
    """掃描所有 pk* 影像，統一報告空位 ID，並在單一視窗中顯示所有結果。"""
    image_folder = 'parking_lot'
    json_folder = 'parking_lot_json'
    
    # 尋找所有以 'pk' 開頭的影像
    image_files = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, f'pk*{ext}')))
        image_files.extend(glob.glob(os.path.join(image_folder, f'pk*{ext.upper()}')))
    
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"在 '{image_folder}' 中找不到任何以 'pk' 開頭的影像檔案。")
        return

    print(f"\n=== 開始掃描 {len(image_files)} 個停車場影像 ===")
    
    all_available_ids = []
    visualized_outputs = []
    detector = EnhancedParkingDetector()

    for image_path in image_files:
        image_basename = os.path.basename(image_path)
        json_filename = os.path.splitext(image_basename)[0] + '.json'
        json_path = os.path.join(json_folder, json_filename)

        if not os.path.exists(json_path):
            print(f"--- 略過 {image_basename}: 找不到對應的 JSON 檔案 ({json_path}) ---")
            continue

        print(f"正在處理: {image_basename}")
        
        if not detector.load_parking_zones(json_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"--- 略過 {image_basename}: 無法讀取影像 ---")
            continue

        result = detector.check_parking_occupancy_enhanced(image)
        
        # 收集空位 ID
        for i, detail in enumerate(result['detection_details']):
            if not detail['is_occupied']:
                zone_id = detector.parking_zones[i].get('id', f'未知ID_{i+1}')
                all_available_ids.append(zone_id)
        
        # 產生並儲存視覺化結果
        output_image = detector.visualize_enhanced_parking(image, result)
        visualized_outputs.append(output_image)

    print("\n=== 掃描完成 ===")

    # 1. 一次性輸出所有空位，並排序
    if not all_available_ids:
        print("\n所有掃描的停車場均無可用空位。")
    else:
        # 定義排序鍵，從 'noid7' 中提取 7
        def sort_key(id_str):
            numbers = re.findall(r'\d+', id_str)
            # 如果找到數字，返回第一個數字的整數值；否則返回一個極大值以便排序在最後
            return int(numbers[0]) if numbers else float('inf')

        # 使用 set 去除重複的 ID，然後排序
        sorted_ids = sorted(list(set(all_available_ids)), key=sort_key)
        print("\n=== 全體空位報告 ===")
        print("找到以下空位 ID:")
        print(', '.join(map(str, sorted_ids)))

    # 2. 將所有視覺化結果排版在同一個視窗顯示
    if visualized_outputs:
        num_images = len(visualized_outputs)
        # 決定縮圖大小
        thumb_w, thumb_h = 800, 600
        
        # 計算網格佈局
        cols = int(math.ceil(math.sqrt(num_images)))
        rows = int(math.ceil(num_images / cols))
        
        # 建立一個黑色畫布
        montage = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
        
        print(f"\n正在準備 {num_images} 張影像的網格預覽...")

        for i, img in enumerate(visualized_outputs):
            # 調整每張圖片大小以符合縮圖尺寸
            resized_img = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            
            # 計算在網格中的位置
            row_idx = i // cols
            col_idx = i % cols
            y_start = row_idx * thumb_h
            x_start = col_idx * thumb_w
            
            # 將縮圖貼到畫布上
            montage[y_start:y_start+thumb_h, x_start:x_start+thumb_w] = resized_img
            
        cv2.namedWindow('All Parking Lots Overview', cv2.WINDOW_NORMAL)
        cv2.imshow('All Parking Lots Overview', montage)
        print("顯示預覽視窗。按任意鍵關閉...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    print("增強型停車檢測系統\n支援多重檢測方法解決車輛遮擋問題\n" + "=" * 60)

    print("\n請選擇操作模式:")
    print("1. 處理單張影像 (詳細視覺化分析)")
    print("2. 掃描所有 'pk' 開頭的影像並回報空位ID")
    
    mode = input("請輸入選項 (1 或 2): ")

    if mode == '1':
        process_single_image()
    elif mode == '2':
        scan_all_for_vacancies()
    else:
        print("無效選項，程式結束。")

if __name__ == "__main__":
    main()