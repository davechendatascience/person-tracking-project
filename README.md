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
*   **多語言說明**：包含繁體中文與技術解析文檔。

## 論文方法解析 (Methodology)

本專案的核心在於實現了 **Follow Everything** 框架中的 **距離幀緩衝區 (Distance Frame Buffer)** 機制。

### 距離幀緩衝區的運作原理：
1. **空間記憶 (Spatial Memory)**：與傳統僅依賴時間序列的追蹤不同，系統會根據領航員與機器人間的**距離 (Depth)** 將高質量的分割特徵（Embeddings/BBoxes）儲存到不同的「距離桶」中。
2. **強健式重新識別 (Robust Re-identification)**：當領航員因為障礙物遮擋或快速移動而丟失時，系統會從緩衝區中提取出與目前領航員距離最相近的歷史影像特徵進行「重新初始化」。
3. **優點**：這能有效解決 SAM2 在目標消失一段時間後，因連續幀間差異過大而無法重新追蹤的問題，大幅提升了機器人在動態環境中的跟隨穩定性。

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

# 指定輸出影片路徑並開啟即時視覺化視窗
python run_video.py --video data/test.mp4 --out-video results/my_result.mp4 --show

# 提升處理速度：每隔一幀取樣（約 2× 吞吐量）
python run_video.py --video data/test.mp4 --frame-stride 2
```

### 3. 可用參數
*   `--video`: 影片檔案路徑。
*   `--output`: 結果輸出目錄（預設：`results/video_mot17_11`）。
*   `--start-frame`: 起始幀索引（預設：`0`）。
*   `--num-frames`: 處理的幀數（預設：`300`）。
*   `--yolo-model`: YOLO 模型路徑（預設：`yolo11m.pt`）。
*   `--target-color`: `red`, `black`, 或 `any`。
*   `--no-reprompt`: 禁用回溯功能（即時模式）。
*   `--max-reprompts`: 設定最大回溯次數（預設：`2`）。
*   `--show`: 開啟即時視覺化視窗（按 `q` 退出）。
*   `--out-video`: 指定輸出影片檔案路徑（預設儲存至 `<output>/tracking_result.mp4`）。
*   `--frame-stride`: 每隔 N 幀處理一幀（預設：`1`）。`--frame-stride 2` 約可使吞吐量翻倍，適合對延遲容忍度較高的場景。

### 4. 效能調校

#### 方法一：Frame Stride（最直接）
```bash
python run_video.py --video data/test.mp4 --frame-stride 2   # ~2× 速度
python run_video.py --video data/test.mp4 --frame-stride 3   # ~3× 速度
```
跳幀追蹤，適合低速移動或離線批次處理場景。

#### 方法二：VOS 優化編譯（初次需 1-3 分鐘預熱）
在 `configs/follow_everything.yaml` 中啟用：
```yaml
sam2:
  vos_optimized: true
```
啟用後，SAM2 的 memory attention、mask decoder 等組件會以 `torch.compile(mode="max-autotune")` 編譯，**第一次執行會觸發長達 1-3 分鐘的編譯**，之後快取於 `~/.cache/torch_extensions`，後續執行速度顯著提升。

## 專案結構說明

*   `follow_everything/perception/sam2_tracker.py`: 核心追蹤邏輯，負責處理緩衝區與 SAM2 呼叫。
*   `run_video.py`: 影片追蹤的主要進入點，包含目標識別與影片匯出邏輯。
*   `configs/follow_everything.yaml`: 追蹤器的超參數配置（信心度閾值、緩衝區大小等）。預設使用 `sam2.1_hiera_tiny.pt` 以平衡速度與精度。

## 引用 (Reference)

如果您在研究中使用此專案，請引用原始論文：

```bibtex
@article{wang2025follow,
  title={Follow Everything: A Leader-Following and Obstacle Avoidance Framework with Goal-Aware Adaptation},
  author={Wang, Youran and others},
  journal={arXiv preprint arXiv:2504.19399},
  year={2025}
}
```

> [!NOTE]
> 更多詳細的技術對照表與實驗結果請參閱 `analysis_results.md`。
