# 論文方法深度解析：Follow Everything

> **論文**：*Follow Everything: A Leader-Following and Obstacle Avoidance Framework with Goal-Aware Adaptation*
> **來源**：arXiv:2504.19399
> **作者**：Wang, Youran et al.

---

## 1. 系統總覽

Follow Everything 是一個針對移動機器人「跟隨領航員」任務設計的端到端框架，由三大模組組成：

```
RGB + Depth ──► [感知模組] ──► leader position p̄ₗ, visibility
                                      │
                              [狀態估計器 EKF]
                              NIS, v̄ₗ, p̂ₗ
                                      │
                              [行為有限狀態機 FSM]
                              state ∈ {Following, Chasing, Retreating, Planning, Switching}
                                      │
                              [拓樸圖規劃器]
                              waypoints
                                      │
                              [速度控制器] ──► (v, ω)
```

硬體平台：Unitree Go2 腿式機器人（Vₘₐₓ = 1.5 m/s）、Realsense D435i、Mid360 LiDAR、Intel i7 + RTX 3070。

---

## 2. 感知模組（§III-A）

### 2.1 SAM2 分割骨幹

論文採用 **EVF-SAM**（基於 SAM2 的變體）進行領航員分割，初始以手動點提示（point prompt）初始化目標。本專案實作改以 **YOLO + 色彩 HSV 過濾** 自動取代手動點提示，完成目標識別後再交由 SAM2 追蹤。

分割結果輸出：語意遮罩 ηₜ 與嵌入特徵 φₜ。領航員 3D 位置由深度對齊點集平均求得：

```
p̄ₗ = mean({ pᵢ = (x, y, z) | pᵢ ∈ segmented point cloud })
```

### 2.2 時間幀緩衝區 ℬᵀ（Temporal Memory Buffer）

維護歷史上信心度最高的前 N 個時間幀嵌入（embeddings），用於增強 SAM2 記憶庫的穩定性：

```
ℬᵀ = top-N { (ηₜ, φₜ, confidence) } sorted by confidence
```

**用途**：當短暫遮擋發生時，以高信心歷史幀補強 SAM2 的記憶，防止追蹤漂移。

本專案實作位置：[`sam2_tracker.py:113-114`](follow_everything/perception/sam2_tracker.py#L113)，`_temp_buf` 列表，`temporal_buffer_size` 控制上限 N。

### 2.3 距離幀緩衝區 ℬᴰ（Distance Frame Buffer）— 核心創新

論文的主要技術貢獻。以機器人與領航員的**相對距離**為索引，每個距離區間（bin）僅保留信心度最高的一幀：

```
bin_idx = floor(depth_m / Δd)
ℬᴰ[bin_idx] = argmax_confidence { (ηₜ, φₜ) } within bin
```

**動機**：人物外觀（視角、尺度）與距離高度相關。當領航員重新出現時，從距離最接近的歷史幀重新提示 SAM2，使其恢復追蹤比從任意歷史幀效果更好。

**重新識別流程（Re-identification）**：
1. 追蹤信心度 < `min_mask_confidence` 時判定目標丟失。
2. 從 ℬᴰ 中選出與 EKF 預測距離最接近的 bin entry。
3. 以該歷史幀的 bbox/mask 重新提示 SAM2，從該時間點重新 propagate。
4. 最多重複 `max_reprompts` 次。

本專案實作位置：[`sam2_tracker.py:536-561`](follow_everything/perception/sam2_tracker.py#L536)，`_update_buffer` 與 `_best_reprompt_entry`。

---

## 3. 狀態估計（§III-B）

### 3.1 卡爾曼濾波器（Kalman Filter）

採用**常速度模型（Constant-Velocity）**的線性卡爾曼濾波器，在機器人坐標系下估計領航員 2D 狀態：

```
狀態向量 x = [px, py, vx, vy]ᵀ

狀態轉移（預測）：
x_{t+1} = F · xₜ
F = [[1, 0, dt, 0],
     [0, 1, 0, dt],
     [0, 0, 1,  0],
     [0, 0, 0,  1]]

觀測模型（僅量測位置）：
z = H · x
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
```

過程雜訊 Q 採用連續白雜訊加速度模型（CWNA），量測雜訊 R 為等向性高斯。

本專案實作位置：[`leader_ekf.py`](follow_everything/estimation/leader_ekf.py)。

### 3.2 Normalized Innovation Squared（NIS）

NIS 是卡爾曼濾波器創新量（innovation）的標準化指標，衡量當前量測與預測的偏差程度：

```
NIS = yₜᵀ · S⁻¹ · yₜ

其中：
  yₜ = zₜ - H · x̂ₜ|ₜ₋₁  （創新量）
  S  = H · Pₜ|ₜ₋₁ · Hᵀ + R （創新共變異數）
```

**語意**：NIS 高 → 濾波器對領航員運動感到「驚訝」→ 不確定性高 → 應拉大跟隨距離。NIS 低 → 預測準確 → 可靠近跟隨。

本專案實作位置：[`leader_ekf.py:103-113`](follow_everything/estimation/leader_ekf.py#L103)。

---

## 4. 行為有限狀態機（§III-B / Fig. 3）

系統以五種狀態描述機器人行為，每幀根據領航員可見性與距離進行轉換：

### 狀態轉換圖

```
              visible & dist < D_min
                  ┌──────────────┐
                  ▼              │
              RETREATING         │
                  │ dist ∈ [D_min, D_max]
                  ▼              │
              FOLLOWING ◄────────┘
               │      │
    dist > D_max│      │not visible
               ▼      ▼
           CHASING  SEARCHING
                        │
              elapsed > search_timeout_s
                        ▼
                     STOPPED

    (SWITCHING：LLM 指令切換領航員，本專案未實作)
```

### 各狀態行為

| 狀態 | 觸發條件 | 目標點 Pᵉⁿᵈ |
|------|----------|-------------|
| **FOLLOWING** | 可見 & D_min ≤ dist ≤ D_max | 以領航員為圓心、半徑 Dₜ 的圓上最近點 |
| **CHASING** | 可見 & dist > D_max | 沿領航員方向距機器人 D_max 處 |
| **RETREATING** | 可見 & dist < D_min | 以領航員為圓心、半徑 D_min 圓上最近點 |
| **SEARCHING** | 不可見 & t < timeout | 上次已知位置 p̄ₗ |
| **STOPPED** | 不可見 & t ≥ timeout | 無（停止） |

本專案實作位置：[`behavior_fsm.py`](follow_everything/control/behavior_fsm.py)。

---

## 5. 控制參數自適應

### 5.1 自適應安全距離

```
Dₜ = Clip(α · NIS, Dᵐⁱⁿ, Dᵐᵃˣ)
```

- `α = nis_alpha = 0.4`（本專案設定）
- 當 EKF 不確定性高（NIS ↑）時，安全距離自動增大，避免碰撞。
- 當預測穩定時，允許靠近至 D_min。

### 5.2 自適應最大速度（Chasing 狀態）

```
Vₜᵐᵃˣ = Clip(α₁ · |v̄ₗ| + α₂ · |p̄ₗ - pᵣ|, 0, Vᵐᵃˣ)
```

- `α₁ = 0.6`：領航員速度權重（越快的領航員，機器人也要越快）
- `α₂ = 0.4`：距離權重（越遠的領航員，機器人加速追趕）
- 上界為物理最大速度 Vᵐᵃˣ = 1.5 m/s

本專案實作位置：[`behavior_fsm.py:221-230`](follow_everything/control/behavior_fsm.py#L221)。

---

## 6. 規劃模組（§III-C）

### 6.1 拓樸圖構建（Topological Graph）

```
輸入：起點 start, 終點 goal, 佔用格地圖 OccupancyGrid
輸出：無碰撞路徑點列表 [w₁, w₂, ..., goal]
```

**流程**：
1. **障礙物群集**：對膨脹後的佔用格地圖做連通域標記（scipy `nd_label`）。
2. **邊界採樣**：對每個群集的凸包（ConvexHull）均勻取樣邊界節點，相鄰節點間距 ≥ `min_boundary_spacing`（0.5 m）。
3. **圖構建**：節點 = {start, goal} ∪ 邊界採樣點；若兩節點間視線（line-of-sight）無遮擋，建立邊（權重 = 歐氏距離）。
4. **Dijkstra 最短路徑**：在圖上求 start → goal 的最短無碰撞路徑。

論文另提出 **Homotopy 剪枝**：生成最多 2ⁿ 條候選路徑（n 為障礙物數），去除同倫類相同的冗餘路徑，本專案以 `max_candidates = 8` 限制。

本專案實作位置：[`topological_graph.py`](follow_everything/planning/topological_graph.py)。

### 6.2 佔用格地圖（Occupancy Grid）

- **解析度**：0.1 m/cell，地圖範圍 12×12 m，以機器人為中心。
- **LiDAR 過濾**：高度 [0.05, 2.0] m 之間的點雲才納入障礙物（排除地板與頭頂）。
- **膨脹（Inflation）**：以機器人半徑 + 安全邊距（0.45 m）做 binary dilation，讓規劃器以點質點處理機器人。

本專案實作位置：[`occupancy_grid.py`](follow_everything/mapping/occupancy_grid.py)。

---

## 7. 實驗結果（論文原始數據）

論文在模擬環境中進行 160 次測試（4 種場景）：

| 指標 | Follow Everything | Alaa et al. | SA-MPC |
|------|:-----------------:|:-----------:|:------:|
| **跟隨成功率** | **96.9%** | 21.8% | 11.9% |
| **目標丟失時間比** | **10.7%** | — | 55.3% |
| **碰撞率** | **1.8%** | — | 80.6% |
| **平均跟隨距離** | **2.0 m** | 3.3 m | 2.4 m |

實體機器人測試場景：室內走廊、室外廣場，領航員包含人類、輪腿機器人。

---

## 8. 本專案實作對照

| 論文模組 | 論文方法 | 本專案實作 | 差異 |
|----------|----------|------------|------|
| 感知初始化 | 手動點提示 → EVF-SAM | YOLO + HSV 色彩過濾 → SAM2 | 自動化；`--target-color random` 可隨機選人 |
| 時間緩衝 ℬᵀ | 論文核心 | 完整實作（`_temp_buf`） | 一致 |
| 距離緩衝 ℬᴰ | 論文核心創新 | 完整實作（`_dist_buf`） | 一致 |
| 狀態估計 | KF（位置+速度） | `LeaderEKF`（filterpy） | 一致 |
| NIS 自適應 | Dₜ = Clip(α·NIS, ...) | `behavior_fsm.py` | 一致 |
| FSM | 5 狀態（含 Switching） | 5 狀態（Switching 未實作） | LLM 切換領航員未納入 |
| 速度自適應 | Vₜᵐᵃˣ = α₁|v̄ₗ| + α₂|dist| | `_chase_speed_scale()` | 一致 |
| 規劃 | 拓樸圖 + Homotopy 剪枝 | 拓樸圖 + Dijkstra | Homotopy 剪枝簡化為候選數上限 |
| 地圖 | LiDAR 佔用格 + 膨脹 | `OccupancyGrid`（scipy） | 一致 |
| 多人追蹤 | 未提及 | `track_sequence_multi`（frame-0 YOLO + SAM2） | 本專案擴充 |

---

## 9. 關鍵設計權衡

1. **距離緩衝 vs. 時間緩衝**：ℬᴰ 的索引是距離而非時間，是因為外觀（appearance）與視角/尺度相關，而尺度由距離決定。純時間緩衝在領航員遠離後重新接近時，歷史幀可能全是不適合的角度。

2. **NIS 驅動距離自適應**：相比固定跟隨距離，NIS 反映了估計不確定性——遮擋、突然轉向時 NIS 升高，機器人自動拉開距離，等濾波器重新收斂後再靠近。這是一種隱式的信心驅動安全機制。

3. **SAM2 vs. 傳統追蹤器**：SAM2 輸出的是語意分割遮罩而非僅 bbox，提供更精確的深度萃取（mask 內中位深度 vs. bbox 均值），且對形變（行走、彎腰）更魯棒。

4. **拓樸圖 vs. 格柵規劃（A*）**：拓樸圖在稀疏障礙場景下速度更快（節點數 << 格柵格數），且能自然地生成繞過障礙物左/右的兩類路徑（Homotopy），而 A* 只返回一條最短路。
