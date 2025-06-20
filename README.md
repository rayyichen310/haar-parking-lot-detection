# 智慧停車格偵測系統 (Intelligent Parking Occupancy Detection)

這是一個基於 OpenCV 與多模態影像分析的智慧停車格偵測系統。傳統的物件偵測方法在面對車輛部分遮蔽或光線不佳等情境時準確率會下降，本專案透過融合多種偵測技術，旨在建立一個更強健、更可靠的停車格佔用狀態判斷引擎。


![image](https://github.com/user-attachments/assets/874bfd4e-81fc-4595-b394-e5b617a96953)


---

## 核心功能

-   **互動式停車格定義**: 提供一個圖形化工具 (`class.py`)，讓使用者能透過滑鼠點擊，輕鬆定義任意形狀（多邊形）的停車格，適應各種停車場佈局。
-   **多模態偵測引擎**: 系統不依賴單一方法，而是融合了三種不同的分析技術來進行交叉驗證：
    1.  **Haar 級聯分類器**: 進行基於特徵的車輛結構偵測。
    2.  **顏色分析 (HSV)**: 分析停車格區域內的顏色分佈，判斷是否有異於地面的物體。
    3.  **邊緣分析 (Canny)**: 計算區域內的邊緣密度，判斷是否存在具有複雜輪廓的物體。
-   **信賴度融合機制**: 將三種方法的結果進行加權計分，產生一個綜合信賴分數，藉此做出更可靠的佔用判斷，有效降低誤判率。
-   **清晰的視覺化結果**: 將偵測結果直接繪製在原始影像上，以不同顏色標示佔用（紅/橘）與空閒（綠）車位，並提供佔用率等即時統計數據。
-   **雙模式操作**:
    -   **單張影像分析**: 對指定的單一影像進行詳細分析與視覺化。
    -   **批次掃描模式**: 快速掃描多張停車場影像，並在終端機統一回報所有空閒車位的 ID。

---


## 專案結構

```
.
├── class.py                  # 互動式停車格定義工具
├── vehicle_detection.py      # 主要的偵測程式
├── requirements.txt          # 專案所需的 Python 套件
│
├── model/
│   └── cascade5.xml          # 預先訓練好的 Haar 分類器模型
│
├── parking_lot/
│   ├── pk1.jpg               # 範例停車場影像
│   └── ...
│
├── parking_lot_json/
│   ├── pk1.json              # 由 class.py 產生的停車格定義檔
│   └── ...
│
└── cascade_opencv_train-master/ # (可選) 用於訓練模型的完整工作區
    │
    ├── 1_labels_to_pos_neg_imgs.py    # 腳本1：從標註檔提取正/負樣本
    ├── 2_generate-negatives.py        # 腳本2：產生負樣本描述檔
    ├── 3_augmentation.py              # 腳本3：對正樣本進行資料增強
    ├── 4_add_aug_positives_to_list.py # 腳本4：產生最終的正樣本描述檔
    ├── positives.info                 # 腳本4產生的最終正樣本描述檔
    │
    ├── dataset/                       # 存放所有原始資料
    │   ├── to_be_annotated/           # 存放待標註的原始圖片
    │   ├── to_be_annotated_xmls/      # 存放 labelImg 產生的 XML 標註檔
    │   ├── pure_positives/            # 存放無需標註的純正樣本圖片
    │   └── pure_negatives/            # 存放不含目標的負樣本背景圖
    │
    └── training_workspace/            # 所有腳本產生的檔案與模型都會存於此
        ├── positives/                 # 存放從 XML 和純正樣本提取的初始正樣本
        ├── aug_positives/             # 存放資料增強後的新正樣本
        ├── neg_bg_from_annotated/     # 存放從已標註圖片中提取的背景圖
        ├── negatives.info             # 腳本2產生的負樣本圖片路徑列表
        ├── positives.vec              # 由 opencv_createsamples 產生的正樣本向量檔
        └── classifier/                # 存放 opencv_traincascade 訓練結果
            ├── cascade.xml            # 最終產生的分類器模型
            ├── stage0.xml, ...        # 訓練過程中的各階段模型
            └── params.xml             # 本次訓練的參數設定
```

---

## 安裝與設定

1.  **複製儲存庫**
    ```bash
    git clone https://github.com/rayyichen310/haar-parking-lot-detection.git
    cd haar-parking-lot-detection
    ```

2.  **安裝相依套件**
    建議先建立一個虛擬環境。本專案的核心相依套件已列在 `requirements.txt` 中。
    ```bash
    pip install -r requirements.txt
    ```

---

## 使用教學

系統的使用分為兩大步驟：先定義停車格，再執行偵測。

### 步驟一：定義停車格 (`class.py`)

此工具會為 `parking_lot` 資料夾中的每一張圖片，產生一個對應的 `JSON` 格式地圖檔，存放在 `parking_lot_json` 資料夾。

1.  **執行定義工具**:
    ```bash
    python class.py
    ```

2.  **選擇影像**: 如果有多張圖片，程式會提示您用數字選擇要處理的圖片。

3.  **繪製停車格**:
    -   **滑鼠左鍵**: 在圖片上沿著一個停車格的邊緣依序點擊頂點（建議至少4個點）。
    -   **滑鼠右鍵**: 完成當前停車格的繪製。
    -   **輸入編號**: 繪製完成後，直接用鍵盤輸入該車位的 ID (例如 `A01`)，然後按下 **Enter** 鍵確認。

4.  **儲存與操作**:
    -   `s` 鍵: 儲存所有已定義的停車格到 `JSON` 檔案。
    -   `u` 鍵: 復原上一個定義好的停車格。
    -   `r` 鍵: 重設所有定義。
    -   `q` 鍵: 退出程式。

### 步驟二：執行偵測 (`vehicle_detection.py`)

此程式會讀取影像和對應的 `JSON` 地圖檔來進行分析。

1.  **執行偵測程式**:
    ```bash
    python vehicle_detection.py
    ```

2.  **選擇模式**:
    -   輸入 `1`: 進入**單張影像分析模式**。程式會讓您選擇要分析的圖片，並顯示詳細的視覺化結果視窗。
    -   輸入 `2`: 進入**批次掃描模式**。程式會自動處理所有以 `pk` 開頭的影像，並在最後於終端機統一列出所有停車場的**可用空位 ID**。

---

## 附錄：模型訓練指令詳解

若您想自行訓練模型，在執行完 `1` 到 `4` 號 Python 腳本後，需使用 OpenCV 提供的工具程式。

**請確保 `opencv_createsamples.exe` 和 `opencv_traincascade.exe` 所在的路徑已加入系統環境變數 `Path` 中。**

### 建立正樣本向量檔 (`opencv_createsamples`)

此指令將 `positives.info` 中列出的所有圖片打包成一個二進位向量檔。

```bash
opencv_createsamples -info positives.info -vec training_workspace/positives.vec -num 5000 -w 60 -h 60
```
-   `-info`: 指定輸入的正樣本描述檔 (`positives.info`)。
-   `-vec`: 指定輸出的 `.vec` 檔案路徑。
-   `-num`: 要產生的正樣本總數。**此數值必須小於或等於 `positives.info` 中的總行數**。
-   `-w`, `-h`: 樣本的寬度和高度（單位：像素），必須與後續訓練時使用的尺寸一致。

### 訓練級聯分類器 (`opencv_traincascade`)

此指令使用產生的 `.vec` 檔案和 `negatives.info` 檔案來進行模型訓練。

```bash
# Windows cmd 使用 ^ 作為換行符
opencv_traincascade -data training_workspace/classifier ^
  -vec training_workspace/positives.vec ^
  -bg training_workspace/negatives.info ^
  -numPos 800 ^
  -numNeg 2000 ^
  -numStages 12 ^
  -w 60 -h 60 ^
  -featureType LBP ^
  -precalcValBufSize 1024 ^
  -precalcIdxBufSize 1024
```
-   `-data`: 指定儲存訓練好的分類器 (`cascade.xml`) 及各階段模型的資料夾。
-   `-vec`: 指定輸入的正樣本 `.vec` 檔案。
-   `-bg`: 指定輸入的負樣本描述檔 (`negatives.info`)。
-   `-numPos`: **每個階段**要使用的正樣本數量。**此數值必須小於 `-num` (來自 `createsamples`)**。
-   `-numNeg`: **每個階段**要使用的負樣本數量。此數值可大於 `negatives.info` 中的圖片總數，因為程式會從中隨機裁切。
-   `-numStages`: 要訓練的分類器總階段數。階段越多，通常越準確，但訓練時間越長。
-   `-w`, `-h`: 訓練視窗的寬度和高度，必須與 `-vec` 檔案的設定完全相同。
-   `-featureType`: 使用的特徵類型。`LBP` (預設) 訓練速度快；`HAAR` 訓練速度非常慢，但可能更準確。
-   `-precalcValBufSize`, `-precalcIdxBufSize`: 分配給預先計算特徵的記憶體大小 (MB)。增加此值可加快訓練速度，但會消耗更多 RAM。

---

## 核心技術

-   **OpenCV**: 用於所有影像讀取、處理、繪圖與 GUI 互動。
-   **NumPy**: 用於高效的數值與陣列運算。
-   **Haar 級聯分類器**: 基於 Viola-Jones 框架的快速物件偵測技術。
-   **Canny 邊緣檢測**: 用於分析影像區域的紋理複雜度。
-   **HSV 色彩空間**: 用於在不同光照條件下，更穩定地分析顏色特徵。
-   **多邊形遮罩 (Masking)**: 透過 `cv2.fillPoly` 和位元運算，精確地隔離出不規則的停車格分析區
