# Follow Everything: 基於 SAM2 的機器人領航員追蹤系統

本專案實作了基於 SAM2 (Segment Anything Model 2) 的強健式視覺追跡開發流程，專為機器人領航（Leader Following）任務設計。本系統結合了 **YOLO 自動目標識別**與 **距離幀緩衝區 (Distance Frame Buffer)** 技術，確保在複雜場景中能穩定地追蹤特定的領航員。

## 核心功能

*   **SAM2 影像追蹤**：利用 Meta 的 SAM2 模型進行高精度的對象分割與追蹤。
*   **自動化目標識別**：整合 YOLO 與 HSV 色彩過濾器，自動識別特定顏色（如：紅色、黑色）的領航員。
*   **距離幀緩衝區 (Distance Frame Buffer)**：參考論文 [Follow Everything (arXiv:2504.19399)](https://arxiv.org/html/2504.19399v1) 實作，利用深度資訊儲存不同距離的高信度特徵。
*   **即時/離線雙模式**：
    *   **離線模式 (Offline)**：進行多輪回溯（Re-propagation）以獲得最佳追蹤路徑。
    *   **在線模式 (Online)**：模擬機器人即時運行的單向追蹤，不進行回溯，適合實際部署。
*   **影片匯出**：自動將追蹤結果合成為 MP4 影片，方便可視化評估。

## 安裝指南

確保您的環境已安裝 CUDA (若有 GPU) 以及相關依賴：

```bash
pip install -r requirements.txt
# 另外需要安裝 SAM2
pip install -e .
```

## 使用方法

### 1. 影片追蹤展示 (強烈建議先進行 H.264 轉檔)

如果您使用的影片格式（如 AV1）無法在 OpenCV 中正常讀取，請先使用 `ffmpeg` 轉編碼：

```bash
ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p output.mp4
```

### 2. 執行追蹤腳本

使用 `run_video.py` 進行自動化追蹤：

```bash
# 基本用法（自動選取畫面中心最明顯的人）
python run_video.py --video output.mp4 --target-color any --num-frames 300

# 追蹤特定顏色（如紅衣人）並開啟即時模式（不回溯）
python run_video.py --video data/test.mp4 --target-color red --no-reprompt

# 手動限制回溯次數（優化長影片處理速度）
python run_video.py --video data/test.mp4 --max-reprompts 1
```

### 3. 可用參數
*   `--video`: 影片檔案路徑。
*   `--target-color`: `red`, `black`, 或 `any`。
*   `--num-frames`: 處理的幀數。
*   `--no-reprompt`: 禁用回溯功能（即時模式）。
*   `--max-reprompts`: 設定最大回溯次數。

## 專案結構說明

*   `follow_everything/perception/sam2_tracker.py`: 核心追蹤邏輯，負責處理緩衝區與 SAM2 呼叫。
*   `run_video.py`: 影片追蹤的主要進入點，包含目標識別與影片匯出邏輯。
*   `configs/follow_everything.yaml`: 追蹤器的超參數配置（信心度閾值、緩衝區大小等）。

## 引用

如果您在研究中使用此專案，請參考：
> Wang et al., "Follow Everything: A Leader-Following and Obstacle Avoidance Framework with Goal-Aware Adaptation," arXiv:2504.19399, 2025.
