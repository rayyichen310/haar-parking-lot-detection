import cv2
import json
import os
import glob
import numpy as np

class ParkingZoneDefiner:
    def __init__(self, image_path=None):
        # 載入影像
        if image_path is None:
            image_path = self.find_image_file()
        if image_path and os.path.exists(image_path):
            self.image = cv2.imread(image_path)
            if self.image is None:
                print(f"無法讀取影像: {image_path}")
                self.image = None
            else:
                print(f"成功載入影像: {image_path}")
                print(f"影像尺寸: {self.image.shape}")
        else:
            print(f"找不到影像檔案: {image_path}")
            self.image = None
        self.zones = []
        self.current_polygon = []
        self.defining_polygon = False
        self.min_points = 3  # 至少3點才能定義多邊形
        # --- 新增狀態變數 ---
        self.is_awaiting_id = False
        self.input_id_str = ""
        self.completed_polygon_points = None

    def find_image_file(self):
        """自動尋找 'parking_lot' 目錄下的影像檔案"""
        input_dir = 'parking_lot'
        if not os.path.isdir(input_dir):
            print(f"警告: 找不到影像目錄 '{input_dir}'。")
            return None
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        for ext in image_extensions:
            files = glob.glob(os.path.join(input_dir, ext)) + glob.glob(os.path.join(input_dir, ext.upper()))
            if files:
                print(f"找到影像檔案: {files}")
                return files[0]
        print(f"在 '{input_dir}' 目錄中找不到任何影像檔案。")
        return None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"左鍵點擊: ({x}, {y})")
            if not self.defining_polygon:
                self.defining_polygon = True
                self.current_polygon = [(x, y)]
                print(f"開始定義停車格 {len(self.zones)+1}")
                print(f"點 1: ({x}, {y})")
            else:
                self.current_polygon.append((x, y))
                print(f"點 {len(self.current_polygon)}: ({x}, {y})")
                if len(self.current_polygon) >= 4:
                    print("已達到4個點，右鍵完成或繼續新增點")
        elif event == cv2.EVENT_RBUTTONDOWN:
            print(f"右鍵點擊: ({x}, {y})")
            if self.defining_polygon and len(self.current_polygon) >= self.min_points:
                # --- 修改：不直接儲存，而是進入等待輸入ID的狀態 ---
                self.completed_polygon_points = self.current_polygon.copy()
                self.is_awaiting_id = True
                self.current_polygon = []
                self.defining_polygon = False
                print(f"完成多邊形定義，共有 {len(self.completed_polygon_points)} 個點。")
                print("請在視窗中輸入車位編號，然後按 Enter 確認。按 Esc 取消。")
            else:
                if not self.defining_polygon:
                    print("目前沒有正在定義的停車格")
                else:
                    print(f"至少需要 {self.min_points} 個點才能完成停車格，目前有 {len(self.current_polygon)} 個點")

    def define_zones(self, json_path=None):
        if self.image is None:
            print("錯誤: 沒有可用的影像")
            return
        window_name = 'Define Parking Spaces Polygon'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        print("\n" + "="*60)
        print("停車格定義工具")
        print("="*60)
        print("操作說明:")
        print("- 左鍵點擊: 新增多邊形頂點")
        print("- 右鍵點擊: 完成當前停車格")
        print("- 按 s: 儲存停車格定義")
        print("- 按 r: 重設所有停車格")
        print("- 按 u: 復原最後一個停車格")
        print("- 按 c: 取消當前正在定義的停車格")
        print("- 按 d: 顯示目前狀態")
        print("- 按 q: 離開")
        print("="*60)
        print("建議: 順時針點擊停車格的四個角點")
        print("="*60)
        while True:
            display_image = self.image.copy()
            # 畫已完成的停車格
            for i, zone in enumerate(self.zones):
                if zone['type'] == 'polygon':
                    points = np.array(zone['points'], dtype=np.int32)
                    cv2.polylines(display_image, [points], True, (0, 255, 0), 6)
                    overlay = display_image.copy()
                    cv2.fillPoly(overlay, [points], (0, 255, 0))
                    cv2.addWeighted(display_image, 0.7, overlay, 0.3, 0, display_image)
                    center_x = int(np.mean([p[0] for p in zone['points']]))
                    center_y = int(np.mean([p[1] for p in zone['points']]))
                    cv2.circle(display_image, (center_x, center_y), 40, (0, 255, 0), -1)
                    # --- 修改：使用 zone['id'] 來顯示編號 ---
                    zone_id = zone.get('id', str(i + 1))
                    font_scale = 1.5
                    # 自動調整字體大小以適應ID長度
                    if len(zone_id) > 3:
                        font_scale = 1.0
                    if len(zone_id) > 5:
                        font_scale = 0.8
                    (text_width, text_height), _ = cv2.getTextSize(zone_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 4)
                    cv2.putText(display_image, zone_id, (center_x - text_width // 2, center_y + text_height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 4)
                    for j, point in enumerate(zone['points']):
                        cv2.circle(display_image, point, 8, (0, 200, 0), -1)
                        cv2.putText(display_image, str(j+1),
                                    (point[0]+15, point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            # 畫正在定義的停車格
            if self.defining_polygon and len(self.current_polygon) > 0:
                for i, point in enumerate(self.current_polygon):
                    cv2.circle(display_image, point, 12, (0, 0, 255), -1)
                    cv2.circle(display_image, point, 18, (255, 255, 255), 4)
                    cv2.putText(display_image, str(i+1),
                                (point[0]+25, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                if len(self.current_polygon) > 1:
                    points = np.array(self.current_polygon, dtype=np.int32)
                    cv2.polylines(display_image, [points], False, (0, 0, 255), 6)
                if len(self.current_polygon) >= 3:
                    points = np.array(self.current_polygon, dtype=np.int32)
                    cv2.polylines(display_image, [points], True, (255, 0, 0), 4)
                    overlay = display_image.copy()
                    cv2.fillPoly(overlay, [points], (255, 0, 0))
                    cv2.addWeighted(display_image, 0.8, overlay, 0.2, 0, display_image)
            # 狀態資訊
            info_y = 50
            # --- 新增：顯示ID輸入提示 ---
            if self.is_awaiting_id:
                prompt_text = f"Enter Space ID: {self.input_id_str}"
                text_size = cv2.getTextSize(prompt_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 4)[0]
                cv2.rectangle(display_image, (20, info_y - 40), (text_size[0] + 40, info_y + 15), (0, 100, 200), -1)
                cv2.putText(display_image, prompt_text, (30, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
                info_y += 60

            status_text = 'Defining' if self.defining_polygon else ('Awaiting ID' if self.is_awaiting_id else 'Waiting to Start')
            hint_text = ("Click the space corners" if self.defining_polygon else
                         ("Enter ID and press Enter" if self.is_awaiting_id else "Left-click to start"))

            info_texts = [
                f"Completed: {len(self.zones)}",
                f"Status: {status_text}",
                f"Points: {len(self.current_polygon)}" if self.defining_polygon else "",
                "Right-click to finish" if len(self.current_polygon) >= 3 else "",
                f"Hint: {hint_text}"
            ]
            for text in info_texts:
                if text:
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 4)[0]
                    cv2.rectangle(display_image, (20, info_y-40), (text_size[0]+40, info_y+15), (0, 0, 0), -1)
                    cv2.putText(display_image, text, (30, info_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
                    info_y += 60
            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(1) & 0xFF

            # --- 新增：處理ID輸入 ---
            if self.is_awaiting_id:
                if ord('0') <= key <= ord('9') or ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z'):
                    self.input_id_str += chr(key)
                elif key == 8:  # Backspace
                    self.input_id_str = self.input_id_str[:-1]
                elif key == 13:  # Enter
                    if self.input_id_str:
                        points = np.array(self.completed_polygon_points)
                        x_coords = [p[0] for p in self.completed_polygon_points]
                        y_coords = [p[1] for p in self.completed_polygon_points]
                        bbox_x = min(x_coords)
                        bbox_y = min(y_coords)
                        bbox_w = max(x_coords) - bbox_x
                        bbox_h = max(y_coords) - bbox_y
                        zone_data = {
                            'id': self.input_id_str,
                            'type': 'polygon',
                            'points': self.completed_polygon_points,
                            'bbox': {
                                'x': bbox_x,
                                'y': bbox_y,
                                'w': bbox_w,
                                'h': bbox_h
                            }
                        }
                        self.zones.append(zone_data)
                        print(f"\n已新增車位，編號: {self.input_id_str}")
                        print(f"  總停車格數: {len(self.zones)}")
                        # Reset state
                        self.is_awaiting_id = False
                        self.input_id_str = ""
                        self.completed_polygon_points = None
                    else:
                        print("\n車位編號不能為空，請重新輸入。")
                elif key == 27: # Escape
                    print("\n已取消新增車位。")
                    self.is_awaiting_id = False
                    self.input_id_str = ""
                    self.completed_polygon_points = None
                continue # 等待下一個按鍵，跳過後面的按鍵處理

            if key == ord('q'):
                break
            elif key == ord('s'):
                if self.zones:
                    save_path = json_path if json_path else 'parking_zones.json'
                    self.save_zones(filename=save_path)
                    print(f"\n已儲存 {len(self.zones)} 個停車格定義到 {save_path}")
                else:
                    print("\n沒有停車格可儲存")
            elif key == ord('r'):
                self.zones = []
                self.current_polygon = []
                self.defining_polygon = False
                print("\n已重設所有停車格")
            elif key == ord('u'):
                if self.zones:
                    removed = self.zones.pop()
                    print(f"\n已復原停車格 {len(self.zones)+1}")
                else:
                    print("\n沒有停車格可復原")
            elif key == ord('c'):
                if self.defining_polygon:
                    self.current_polygon = []
                    self.defining_polygon = False
                    print("\n已取消當前停車格定義")
                else:
                    print("\n目前沒有正在定義的停車格")
            elif key == ord('d'):
                print(f"\n目前狀態:")
                print(f"   總停車格: {len(self.zones)}")
                print(f"   正在定義: {self.defining_polygon}")
                print(f"   目前點數: {len(self.current_polygon)}")
                if self.current_polygon:
                    print(f"   目前點座標: {self.current_polygon}")
                for i, zone in enumerate(self.zones):
                    print(f"   停車格 {i+1}: {len(zone['points'])} 個點")
        cv2.destroyAllWindows()
        if self.zones:
            print(f"\n完成，總共定義了 {len(self.zones)} 個停車格")
            for i, zone in enumerate(self.zones):
                print(f"   停車格 {i+1}: {zone['points']}")
        else:
            print("\n沒有定義任何停車格")

    def save_zones(self, filename='parking_zones.json'):
        data = {
            'zones': self.zones,
            'total_spaces': len(self.zones),
            'zone_type': 'polygon',
            'created_at': str(cv2.getTickCount()),
            'image_shape': self.image.shape if self.image is not None else None
        }
        try:
            output_dir = os.path.dirname(filename)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            with open(filename, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                print(f"檔案包含 {saved_data['total_spaces']} 個停車格")
        except Exception as e:
            print(f"儲存時發生錯誤: {e}")

def main():
    print("停車格定義工具")
    print("支援多邊形停車格與詳細除錯資訊")
    print("=" * 60)

    input_dir = 'parking_lot'
    output_dir = 'parking_lot_json'

    if not os.path.isdir(input_dir):
        print(f"錯誤: 找不到輸入目錄 '{input_dir}'")
        print("請建立 'parking_lot' 目錄並將影像放入其中。")
        os.makedirs(input_dir, exist_ok=True)
        print(f"已建立 '{input_dir}' 目錄。")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- 修改：使用 set 來避免重複的檔案 ---
    image_files_set = set()
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    for ext in extensions:
        # 在不分大小寫的系統(如Windows)上，glob會找到所有符合的檔案
        # 使用 set 可以確保即使在分大小寫的系統上，大小寫不同的同名檔案也不會重複
        for file_path in glob.glob(os.path.join(input_dir, ext)):
            image_files_set.add(file_path)
        for file_path in glob.glob(os.path.join(input_dir, ext.upper())):
            image_files_set.add(file_path)
    
    # 將 set 轉換為排序過的 list，確保順序一致
    image_files = sorted(list(image_files_set))

    if not image_files:
        print(f"在 '{input_dir}' 目錄找不到影像檔案")
        print("請將停車場影像放置在該目錄")
        return

    print("找到以下影像檔案:")
    for i, file in enumerate(image_files):
        print(f"{i+1}. {os.path.basename(file)}")

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
    output_json_path = os.path.join(output_dir, json_filename)

    print(f"\n將處理影像: {selected_image}")
    print(f"JSON 儲存路徑: {output_json_path}")

    definer = ParkingZoneDefiner(selected_image)
    definer.define_zones(json_path=output_json_path)

if __name__ == "__main__":
    main()