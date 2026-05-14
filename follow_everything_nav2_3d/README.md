# follow_everything_nav2_3d

[`follow_everything_nav2`](../follow_everything_nav2/) 的 3D Gazebo Fortress + ROS 2 Humble 版本，把 2D 模擬中的 oracle 相機換成**真正的 RGB-D 視覺追蹤器**：預設使用 [EdgeTAM](../EdgeTAM/)（SAM2 的輕量版分支，~20 Hz），可切換為 [AOT/DeAOT](https://github.com/yoxu515/aot-benchmark)（具長期記憶、抗遮擋的 streaming VOS）。

Topic contract 與 2D 版本完全相同（行為樹原樣搬移）：

| topic                            | type                          | 方向            |
|----------------------------------|-------------------------------|------------------|
| `/follower/odom`                 | `nav_msgs/Odometry`           | sim → follower   |
| `/follower/scan`                 | `sensor_msgs/LaserScan`       | sim → follower   |
| `/follower/camera/detections`    | `vision_msgs/Detection2DArray`| sim → follower   |
| `/follower/cmd_vel`              | `geometry_msgs/Twist`         | follower → sim   |

---

## 1. 系統需求與依賴

| 項目 | 版本 / 條件 |
|------|------------|
| 主機 OS | Ubuntu 22.04（Jammy）arm64 / amd64 |
| GPU | NVIDIA，已安裝 CUDA 13.0 driver（可向下相容 cu12.x runtime） |
| **主機 CUDA toolkit** | **`/usr/local/cuda-13.0`** 必須存在（`spatial_correlation_sampler` 在容器內編譯時會掛這個路徑） |
| Docker | 20.10+ |
| Compose | v2（`docker compose ...`） |
| NVIDIA Container Toolkit | 已裝（`--gpus` / `deploy.resources.reservations.devices`） |
| 顯示 | X server（本機）或 SSH X11 forwarding / VNC（遠端） |

確認主機端 CUDA toolkit：

```bash
ls /usr/local/cuda-13.0/bin/nvcc          # 必須存在
nvidia-smi                                # driver 跑得起來
```

---

## 2. 目錄結構

```
follow_everything_nav2_3d/
├── Dockerfile, docker-compose.yml
├── sim/                              # sim 端：世界、URDF、oracle、leader
│   ├── worlds/empty.world             # 內含 follower / leader 的 inline SDF
│   ├── python/
│   │   ├── oracle_camera.py           # ground truth 相機（AOT bootstrap + 對照組）
│   │   ├── leader_controller.py       # A* random-goal patrol，驅動 /leader/cmd_vel
│   │   ├── world_odom_publisher.py    # /gz_pose_truth → /follower/odom（世界座標系）
│   │   ├── lidar_leader_filter.py     # 從 /follower/scan_raw 移除 leader 自身回波
│   │   ├── snapshot_recorder.py       # 每秒輸出一張俯視 PNG
│   │   └── build_world.py             # 從 2D ASCII 地圖生成 3D 世界
│   └── launch/empty_bringup.launch.py
├── follower_pkg/                     # follower 端：追蹤器、BT、follower launch
│   ├── python/
│   │   ├── edgetam_tracker.py         # SAM2-fork streaming tracker（預設）
│   │   ├── aot_tracker.py             # AOT/DeAOT streaming tracker
│   │   └── simple_follower.py         # P-controller（regression 用）
│   └── launch/follower.launch.py
└── eval/                             # 端到端評估腳本
    ├── record_episode.py              # 一鍵啟動整套，固定時長後關閉
    ├── smoke_tracker_aot.py           # 離線 AOT 煙霧測試（不需要 ROS / Gazebo）
    └── smoke_edgetam*.py              # EdgeTAM 對照組煙霧測試
```

---

## 3. 技術堆疊

- **ROS 2 Humble**
- **Ignition Gazebo Fortress**（LTS，與 Humble 為 Tier 1；改用 Fortress 是因為 arm64/Jetson 上沒有 `gazebo_ros` Classic-11 套件）
- `ros_gz_bridge` 雙向映射 Gazebo Transport ↔ ROS topic
- **追蹤器**：EdgeTAM（預設）或 AOT/DeAOT（CVPR 2022 / 2023）
- **感測器**：RGB-D 相機（640×480 @ 20 Hz、90° H-FOV、0.1–10 m）+ 360° lidar（72 線、5°、8 m、20 Hz）
- **機器人**：差速驅動、車身半徑 0.25 m、最大線速度 1.5 m/s、最大角速度 1.5 rad/s

---

## 4. 顯示 / X11 設定

Gazebo GUI、RViz、EdgeTAM / AOT overlay 都需要容器內能存取 X server。
[`docker-compose.yml`](docker-compose.yml) 已掛載 `/tmp/.X11-unix` 並設好 `DISPLAY` / `XAUTHORITY`，差別只在於主機端如何允許容器連入。

### 本機 Linux 主機

```bash
xhost +local:root              # 允許容器內 root 繪圖
touch /tmp/.docker.xauth       # compose 掛載點（不能是不存在的檔案）
```

### 透過 SSH 從遠端工作站連入

```bash
# 在工作站
ssh -X -C user@gb10            # -X forwarding、-C 壓縮；-X 被拒就改 -Y
```

進到 GB10 之後：

```bash
echo "$DISPLAY"                                # 應該是 localhost:10.0 之類
xhost +local:root
touch /tmp/.docker.xauth
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -
```

X11 forwarding 適合 RViz / image stream，跑 Gazebo 3D 視窗會偏慢；要流暢的 Gazebo 建議在 GB10 上跑 VNC / NoMachine。

### 常見問題

| 症狀                                                                  | 排查方向                                                              |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| `Authorization required, but no authorization protocol specified`     | 在同一 display 的 shell 重跑 `xhost +local:root`                       |
| `cannot open display:`                                                | `DISPLAY` 沒匯出或 `/tmp/.X11-unix` 沒掛上                              |
| Gazebo 開起來是黑窗 / "failed to create drawable"                      | 容器內沒有 OpenGL context，X11 forwarding 試 `LIBGL_ALWAYS_INDIRECT=1` |
| RViz / overlay 沒問題，Gazebo 不行                                     | Gazebo 需要 direct GL，X11 forwarded GL 不夠力，改用 VNC                |

---

## 5. 啟動 EdgeTAM（預設追蹤器，最簡路徑）

```bash
cd follow_everything_nav2_3d
docker compose build
docker compose up -d sim

# 終端 1 — Fortress + spawn follower + bridges
docker exec -it follow_everything_nav2_3d bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 launch sim/launch/empty_bringup.launch.py'

# 終端 2 — EdgeTAM tracker + BT follower
docker exec -it follow_everything_nav2_3d bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 launch follower_pkg/launch/follower.launch.py'
```

切換感測來源（EdgeTAM 為預設、oracle 為對照組）：

| 命令                                                                          | `/follower/camera/detections` 由誰發布          |
|------------------------------------------------------------------------------|-----------------------------------------------|
| `ros2 launch sim/launch/empty_bringup.launch.py` *（預設）*                   | EdgeTAM（oracle 改名為 `_oracle`）               |
| `ros2 launch sim/launch/empty_bringup.launch.py detection_source:=oracle`     | oracle 直接發布（EdgeTAM 改名為 `_edgetam`）       |

切換地圖：將 `empty` 改為 `cluttered`、`forest`、`corridor`；`build_world.py` 會從 2D 專案的 ASCII 地圖（`../follow_everything_nav2/sim/maps/*.txt`）自動生成對應的 3D 世界。

---

## 6. 切換為 AOT/DeAOT 追蹤器

AOT/DeAOT 的整合路徑分成四個階段：**(a) 取得 aot-benchmark 原始碼與權重 → (b) 套用我們改過的 demo.py → (c) 用 demo.py 做離線驗證 → (d) 在容器內安裝 CUDA correlation kernel → (e) 切換 record_episode 為 AOT**。每一步都建議先過再進下一步。

### 6.1 取得 aot-benchmark 原始碼

aot-benchmark 並非本 repo 的 submodule，需要 clone 到**本 repo 同層**目錄（compose 會以 `../aot-benchmark` 路徑掛進容器的 `/opt/aot-benchmark`）：

```bash
# 在 person-tracking-project/ 同層執行（不是這個子目錄內）
cd ..                          # 確認位於 person-tracking-project 根目錄
git clone https://github.com/yoxu515/aot-benchmark.git
```

確認結果：

```bash
ls aot-benchmark/tools/demo.py            # 必須存在
ls aot-benchmark/networks/engines/        # aot_engine.py / deaot_engine.py 必須存在
```

### 6.2 下載預訓練權重

aot-benchmark 的權重托管在 Google Drive（見 [`aot-benchmark/MODEL_ZOO.md`](../aot-benchmark/MODEL_ZOO.md)）。本專案預設使用 **R50-DeAOTL（PRE_YTB_DAV stage）**，建議至少準備這一個檔案；若要 CPU 友善的最小變體，再準備一個 DeAOTT：

| 模型           | 檔名                                | Param | Google Drive                                                                                |
|---------------|------------------------------------|------|----------------------------------------------------------------------------------------------|
| **R50-DeAOTL** | `R50_DeAOTL_PRE_YTB_DAV.pth`       | 19.8 M | [link](https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/view?usp=sharing) |
| DeAOTT        | `DeAOTT_PRE_YTB_DAV.pth`           |  7.2 M | [link](https://drive.google.com/file/d/1ThWIZQS03cYWx1EKNN8MIMnJS5eRowzr/view?usp=sharing) |
| SwinB-DeAOTL  | `SwinB_DeAOTL_PRE_YTB_DAV.pth`     | 70.3 M | [link](https://drive.google.com/file/d/1g4E-F0RPOx9Nd6J7tU9AE1TjsouL4oZq/view?usp=sharing) |

下載後放到 `aot-benchmark/pretrain_models/`：

```bash
ls aot-benchmark/pretrain_models/
# R50_DeAOTL_PRE_YTB_DAV.pth
# DeAOTT_PRE_YTB_DAV.pth        （可選）
# SwinB_DeAOTL_PRE_YTB_DAV.pth  （可選）
```

### 6.3 套用對 `tools/demo.py` 的修改（**必要**）

vanilla `tools/demo.py` 在 1001 幀的長影片上會在第 ~355 幀因 **CUDA 記憶體碎片化（不是 VRAM 用光）** 而 OOM。我們對 `demo.py` 做了五項擴充，現在仍以 uncommitted diff 的形式存在於 `aot-benchmark/tools/demo.py`（執行 `git diff -- tools/demo.py` 可看）：

| 修改 | 為什麼 |
|------|--------|
| **`PYTORCH_CUDA_ALLOC_CONF`**（檔頭，import torch 前）：`max_split_size_mb:128,garbage_collection_threshold:0.7,expandable_segments:True` | 必須在 `import torch` **之前**設定，因為 PyTorch allocator 只在初始化時讀一次。`expandable_segments:True` 把碎片化的 reserved 區段還給 OS，1001 幀跑完碎片率從 4.2% 收斂到 2.1%。 |
| **`_install_lt_cap()` + monkey-patch `AOTEngine.update_long_term_memory`**：批次驅逐 long-term memory bank | vanilla DeAOT 每 `TEST_LONG_TERM_MEM_GAP` 幀就在 dim=0 上 concat 一筆且**永不淘汰**，造成 OOM。我們改成達到 `lt_max` 後每 `batch_evict_every` 幀保留最新的 `lt_max × keep_ratio` 筆；批次驅逐對 allocator 比逐幀切片友善很多。 |
| **`cache_clear_every`**：每 N 幀執行 `gc.collect() + torch.cuda.empty_cache()` | 主動釋放空閒 segment，避免 fragmentation 越累積越大。 |
| **`frag_log_every`**：定期 print `allocated / reserved / frag` 量測值 | 除錯時可看碎片化收斂曲線。 |
| **`conf_thresh`**（CLI 啟用，預設關）：mean softmax probability gate | 我們的測試片段中 distractor 也有高信度，**沒有幫上忙**，保留是 opt-in；要驗證自己的 dataset 時可用。 |

新增的 CLI 旗標（每個都有 default，省略時等同 vanilla 行為，但建議照下方推薦值跑）：

| flag                  | 推薦              | 說明                                  |
|----------------------|------------------|---------------------------------------|
| `--lt_max`            | `80`             | long-term memory bank 硬上限（幀數）   |
| `--lt_batch_evict`    | `50`             | 每 N 幀檢查並批次驅逐                  |
| `--lt_keep_ratio`     | `0.8`            | 驅逐時保留最新的 `lt_max × ratio`     |
| `--lt_gap`            | `5`–`10`         | 覆寫 `TEST_LONG_TERM_MEM_GAP`         |
| `--cache_clear_every` | `200`            | 0 = 停用                              |
| `--frag_log_every`    | `200`            | 0 = 停用                              |
| `--conf_thresh`       | `0.0`            | 0 = 停用（distractor 多的場景不要開） |

如果你 clone 出來的 `aot-benchmark/tools/demo.py` 還是 vanilla 版，可以：

```bash
cd aot-benchmark
git diff --stat tools/demo.py   # 看是不是 vanilla（應該 0 行 modified）

# 取得我們的修改：直接從本 repo cherry-pick uncommitted diff
# （這份修改目前仍以 uncommitted 形式存在，未送 PR 上游）
cd /path/to/person-tracking-project/aot-benchmark
# 若本機已有 working tree 改動：保留之
# 若是新 clone：把上游 fork 的 demo.py 換成我們的版本即可
```

> 註：我們也計畫把上述修改整理成上游 PR，現階段請以本 repo 的 working tree 為準。

### 6.4 用 demo.py 做離線驗證（**強烈建議**）

在進到容器整合前，先用 vanilla CLI 驗證：(a) 權重能載入、(b) inference 跑得起來、(c) 改過的 long-term memory cap 確實生效（看到 `[LT-cap] batch evict` log）。

aot-benchmark/demo.py 的標準呼叫慣例（影像幀序列 + 第一幀 mask）：

```bash
cd aot-benchmark
python tools/demo.py \
  --model r50_deaotl \
  --stage pre_ytb_dav \
  --ckpt_path pretrain_models/R50_DeAOTL_PRE_YTB_DAV.pth \
  --data_path datasets/your_video/JPEGImages/ \
  --output_path /tmp/aot_demo_out \
  --lt_max 80 --lt_batch_evict 50 --lt_keep_ratio 0.8 --lt_gap 5 \
  --cache_clear_every 200 --frag_log_every 200
```

期望輸出：

```text
[LT-cap] enabled: lt_max=80, batch_evict_every=50, keep_ratio=0.8
[allocator] PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.7,expandable_segments:True
[mem] frame=200 alloc=5810.3MB reserved=5942.4MB frag=2.2%
[LT-cap] batch evict (frame=400, kept=64, max=80, total_evicts=4)
...
```

若 VRAM 不足或想跑無 ROS / 無 Gazebo 的最快煙霧測試，**改用本 repo 的離線腳本**（mp4 直接吃，內含 YOLO-seg 自動取第一幀 mask）：

```bash
# 容器內，從 /ws：
python3 eval/smoke_tracker_aot.py path/to/input.mp4
AOT_MODEL=deaott python3 eval/smoke_tracker_aot.py path/to/input.mp4
```

煙霧測試會印 `enable_corr (CUDA correlation kernel)=False — pure-PyTorch fallback`，這代表還沒裝 CUDA correlation kernel（下一步要做），但 inference 仍能跑（會慢 ~3–5×）。

### 6.5 在容器內安裝 `spatial_correlation_sampler` CUDA kernel（**一次性**）

AOT 的 matching attention fast path 需要 `spatial_correlation_sampler` 這支 C++ CUDA 擴充。映像檔中**故意不包含**這個套件（也不裝 CUDA toolkit）：理由是 NVIDIA CDN DNS 容易掉、aarch64 SBSA 平台需要的 500 MB toolkit 包很容易讓 image build 卡住，所以改採「**主機 CUDA toolkit 唯讀掛載 + 容器內首次執行時 pip 編譯 + docker commit 烘進 image**」三步驟。

主機端 CUDA toolkit 的路徑在 [`docker-compose.yml`](docker-compose.yml) 已掛好：

```yaml
volumes:
  - /usr/local/cuda-13.0:/usr/local/cuda-13.0:ro
```

Dockerfile 也已預先設好 `CUDA_HOME` / `PATH` / `LD_LIBRARY_PATH` 指向這個 mount。所以**容器一跑起來，nvcc 就在 `$PATH` 中可用**。執行：

```bash
docker compose up -d sim

# 容器內第一次裝（編譯約 1–2 分鐘）
docker exec follow_everything_nav2_3d \
    pip install --no-cache-dir --no-build-isolation spatial-correlation-sampler

# 驗證 import 成功
docker exec follow_everything_nav2_3d python3 -c \
    "from spatial_correlation_sampler import SpatialCorrelationSampler; print('OK')"
```

把編譯結果烘進 image，避免每次 `docker compose up` 都要重裝：

```bash
docker commit follow_everything_nav2_3d follow_everything_nav2_3d:latest
```

> ⚠ `docker commit` 寫進去的 layer **會被 `docker compose build` 覆蓋**。下次 rebuild image 後要重做 6.5 一次。

### 6.6 用 AOT 跑 `record_episode.py`

```bash
# 容器內，從 /ws：
source /opt/ros/humble/setup.bash

# 60 秒 AOT/DeAOT 跑 forest 地圖
TRACKER_KIND=aot python3 eval/record_episode.py 60 edgetam forest
```

說明：第二個 positional 參數 `edgetam` 是 legacy 名稱、意思是「**追蹤器**負責發布 contract topic」，**不是**指定哪一個追蹤器二進位；追蹤器選擇來自 `TRACKER_KIND` 環境變數。

正常啟動會看到 follower log 中：

```text
AOT tracker model=r50_deaotl stage=pre_ytb_dav ckpt=/opt/aot-benchmark/pretrain_models/R50_DeAOTL_PRE_YTB_DAV.pth
AOT enable_corr (CUDA correlation kernel)=True — fast path
AOT memory: LT_GAP=5 (store every Nth frame), LT_MAX=80 (cap), BATCH_EVICT_EVERY=50 frames, KEEP_RATIO=0.8
Installed batched-eviction long-term memory cap on DeAOTEngine.update_long_term_memory
AOT predictor built in 5.2s
AOT init: mask shape=(480, 640) px=...
```

若 `enable_corr=...=False — pure-PyTorch fallback` 表示 6.5 沒完成（kernel 沒裝、或 image 被 rebuild 後 `docker commit` 沒重做），AOT 仍會跑，只是會慢 ~3–5×。

#### 切換 AOT 模型 / 權重

```bash
# 用較大的 SwinB-DeAOTL（accuracy 較高，VRAM 較吃）
TRACKER_KIND=aot \
AOT_MODEL=swinb_deaotl \
AOT_CKPT=/opt/aot-benchmark/pretrain_models/SwinB_DeAOTL_PRE_YTB_DAV.pth \
  python3 eval/record_episode.py 120 edgetam cluttered
```

AOT 環境變數（皆可選，列出預設值；皆於 1001 幀測試影片上驗證過：穩定 5.8 GB VRAM、最終碎片率 2.1%、無 OOM）：

| 環境變數                    | 預設                                  | 說明 |
|---------------------------|--------------------------------------|------|
| `AOT_MODEL`               | `r50_deaotl`                         | `aot-benchmark/configs/models/` 下的 model config |
| `AOT_STAGE`               | `pre_ytb_dav`                        | training stage（`pre` / `pre_ytb_dav` / `pre_ytb`） |
| `AOT_CKPT`                | 由 `AOT_MODEL` 自動推導               | checkpoint 路徑覆寫 |
| `AOT_LT_GAP`              | `5`                                  | 每 N 幀寫一筆 long-term memory |
| `AOT_LT_MAX`              | `80`                                 | long-term memory 條目硬上限 |
| `AOT_LT_BATCH_EVICT_EVERY`| `50`                                 | 每 N 幀檢查並批次驅逐（只有超過 `AOT_LT_MAX` 才驅逐） |
| `AOT_LT_KEEP_RATIO`       | `0.8`                                | 驅逐時保留最新 `AOT_LT_MAX × ratio` 筆 |
| `AOT_MAX_LONG_EDGE`       | `800`                                | 輸入影像長邊像素上限（resize cap） |
| `AOT_DEBUG_FRAMES`        | `8`                                  | 啟動時 dump 前 N 幀 init RGB + propagated mask 到 `EP_LOG_DIR/` |
| `TRACKER_TASKSET_CORES`   | 未設                                 | 給追蹤器 process 的 cgroup mask（例如 `"0,1"`） |
| `FOLLOWER_TASKSET_CORES`  | 未設                                 | 給 BT follower 的 cgroup mask（與 tracker 互補） |

---

## 7. `eval/record_episode.py` — 一鍵端到端評估

啟動順序：
1. **WORLD**：gz Fortress + 三個 ros_gz_bridge + `world_odom_publisher` + `lidar_leader_filter` + `snapshot_recorder`
2. **LEADER**：`oracle_camera`
3. **追蹤器**：依 `TRACKER_KIND` 啟動 `edgetam_tracker.py` 或 `aot_tracker.py`
4. **阻塞等待**：偵測 follower.log 中出現 `AOT init: mask shape=` 或 `EdgeTAM init: mask shape=` 才繼續，避免追蹤器尚未鎖定就被 leader 走掉
5. **LEADER patrol**：`leader_controller.py`（A* random-goal patrol）
6. **FOLLOWER**：2D 專案掛載過來的 BT-based `follow_everything_follower.py`
7. 跑 `duration_sec` 秒後送 SIGINT、SIGTERM 收掉所有 process group

```bash
python3 eval/record_episode.py [duration_sec] [detection_source] [map]
```

| 參數                | 可能值                                  | 預設     |
|--------------------|-----------------------------------------|----------|
| `duration_sec`     | 整數秒                                   | `30`     |
| `detection_source` | `oracle` \| `edgetam`                   | `oracle` |
| `map`              | `empty` \| `cluttered` \| `corridor` \| `forest` | `empty`  |

範例：

```bash
# 90 秒 oracle 跑 empty 地圖（baseline / 對照組）
python3 eval/record_episode.py 90 oracle empty

# 90 秒 EdgeTAM 真實感知跑 cluttered 地圖
python3 eval/record_episode.py 90 edgetam cluttered

# 60 秒 AOT 真實感知跑 forest 地圖（最考驗追蹤器的場景）
TRACKER_KIND=aot python3 eval/record_episode.py 60 edgetam forest
```

輸出結構：

```
results/logs/ep_<ts>_<map>_0/
├── world.log        # gz + bridges + world_odom_publisher + lidar_leader_filter
├── leader.log       # oracle_camera + leader_controller
├── follower.log     # 追蹤器 + follow_everything_follower (BT)
├── snapshots.log    # snapshot_recorder
└── snapshots/       # 每秒一張俯視 PNG（pose、FOV、A* 路徑、last_seen）
```

從容器外觸發：

```bash
docker exec follow_everything_nav2_3d bash -lc \
  'source /opt/ros/humble/setup.bash && cd /ws && \
   TRACKER_KIND=aot python3 eval/record_episode.py 60 edgetam forest'
```

---

## 8. 從錯誤中學到的工程經驗（精選）

- **「長 inference 迴圈裡的 CUDA OOM 八成是碎片化、不是 VRAM 滿。」** 先 `print(memory_allocated / memory_reserved)`；碎片率高（>20%）就加 `expandable_segments:True` + 批次驅逐通常就解決，不需動模型。
- **不要在 image 內裝 CUDA toolkit**（特別是 sbsa aarch64）：~500 MB + 對 NVIDIA CDN DNS 敏感。改用主機掛載 `/usr/local/cuda-XX.X` + 容器內一次性 `pip install` + `docker commit`。
- **`docker commit` 寫進去的 layer 會被 `docker compose build` 覆蓋**：rebuild 後要重做 6.5。
- **批次驅逐 vs 逐幀切片**：對 CUDA allocator 而言批次（每 50 幀一次切大區段）比逐幀（每幀切薄薄一條）友善得多，碎片率收斂方向相反。
- **Gazebo Fortress 的 `<actor>` 不接受 `<link><collision>`**：是 SDF 規範陷阱（未明確記錄）。要 leader 帶 walk animation，要用 actor + scripted trajectory，但會失去 collision，跟 lidar 串接得另想辦法（目前 leader 是 kinematic SDF model + VelocityControl plugin，沒走 actor）。

---

## 9. 已知議題

- **遮擋恢復**：EdgeTAM / AOT 都是單目追蹤器，leader 被遮住時短時間內 BT 的 SweepRecover 會接手；長時間遮擋下會永久失鎖。目前沒有自動 re-init 邏輯。
- **Streaming 暫存目錄**：EdgeTAM tracker 每幀寫一張 JPEG 到 `/tmp/edgetam_stream_*`，長時間執行需手動清理或重啟容器。
- **里程計沒有雜訊模型**：目前 odom 來自 `world_odom_publisher`（gz SceneBroadcaster ground truth）。下一階段會加上 EKF + latency + TF 清理。
- **conf_thresh 在多 distractor 場景沒幫上忙**：實測 distractor 也常有高信度（forest 場景下尤其明顯），預設關閉；保留是 opt-in flag 給你自己的 dataset 試。
