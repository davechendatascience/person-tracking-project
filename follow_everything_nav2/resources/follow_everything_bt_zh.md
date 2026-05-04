# Follow-Everything：以行為樹（Behavior Tree）實作領隨者追蹤

本文件以繁體中文整理整套追蹤演算法在 py\_trees（與 Nav2 BT 同一個家族）上的設計。其核心理念是：**以優先序選擇器（Selector）排出 6 個葉節點，每個 tick（20 Hz）由最高優先且仍可運作的葉節點接管 `cmd_vel`**。沒有狀態機、沒有顯式轉移，靠葉節點自身的 `SUCCESS / FAILURE / RUNNING` 自動裁決。

---

## 1. 感知資料來源（嚴格無洩漏）

| Topic | 內容 | 角色 |
| --- | --- | --- |
| `/follower/odom` | 跟隨者自身 pose | 自我定位 |
| `/follower/scan` | 2D LiDAR 360° | 即時建圖 |
| `/follower/camera/detections` | 領導者 body-frame 位置（僅當可見） | 唯一的領導者資訊來源 |
| `/follower/camera/pedestrians` | 行人 body-frame 位置 | 動態障礙避讓 |

**禁止訂閱** `/leader/*` 任何 ground-truth topic、**禁止載入靜態地圖**。地圖大小（如 15 × 15 m）視為部署設定，可作為超參數，但障礙物完全靠 LiDAR 學出來。

---

## 2. 線上建圖：log-odds 占據格

```python
LOG_ODDS_HIT, LOG_ODDS_MISS = +0.4, -0.4
LOG_ODDS_CLAMP = 2.0
LOG_ODDS_THRESH_BLOCKED = 0.3   # > 此值 = 確定有牆
```

每筆 LiDAR scan 對每條光束沿線上的 cell 加 `MISS`，到端點的 cell 加 `HIT`，並夾在 `±CLAMP` 之內。動態行人因為走過後又被光束穿透回 `MISS`，會在 ~0.25 秒內淡出；靜態牆則每幀都被打到 `HIT`，永遠保持高機率。

**領導者特例**：跟隨者長時間貼著領導者，光束會把領導者誤判為永久障礙。解法是用相機觀察到的方位角開一個錐形遮罩，在該方位的 LiDAR 不更新 occupancy。這是針對「領導者」這個特定目標的合理感測融合，不是地圖洩漏。

### 2.1 從 log-odds 產生規劃用 grid

關鍵踩過的雷：`merged_aabbs_from_grid` 預設使用影像座標（row 0 = 世界頂部，`ymax = (H-j) × cell`），但我們的 `log_odds[j, i]` 是 row j → `y = j × MAP_RES`（不翻轉）。直接餵進去會把所有觀察到的牆**沿世界 Y 中軸鏡射**，於是 A\* 在「幻影自由空間」裡規劃路徑、實際撞牆。

修正：

```python
binary = log_odds > LOG_ODDS_THRESH_BLOCKED
aabbs = merged_aabbs_from_grid(np.flipud(binary), MAP_RES)

# 同時用形態學擴張（disk 結構元素）
# 把每個 LiDAR 端點 cell 直接膨脹成圓盤，
# 1 cell 厚的薄牆也不會被對角線 A* 鑽過去。
r_cells = ceil(INFLATE_RADIUS / PLAN_RES)   # 0.4 / 0.2 = 2
yy, xx = np.ogrid[-r_cells:r_cells+1, -r_cells:r_cells+1]
struct = (xx*xx + yy*yy) <= r_cells*r_cells
plan_grid = binary_dilation(binary, structure=struct)
```

接著在 `plan_grid` 上用 EDT（Euclidean distance transform）算每個自由 cell 到最近障礙的距離，反推出 cost 加成（離牆愈近成本愈高），交給加權 A\*。這樣路徑會自然偏好寬敞走廊。

---

## 3. 行為樹結構

```python
root = Selector(name="follow_everything_v34", memory=False)
root.add_children([
    Retreating("Retreating", bb),           # 太近就退
    Following("Following", bb),             # 看得見且距離適中
    Chasing("Chasing", bb),                 # 看得見但太遠
    PlannedRecovery("PlannedRecovery", bb), # 看不見：A* 去 last_seen
    BackupRecovery("BackupRecovery", bb),   # 還沒到 last_seen 的後備
    SpiralExpand("SpiralExpand", bb),       # 到了 last_seen 還是看不見，環狀搜索
])
```

`memory=False` 是關鍵：**每個 tick 都從頭重新評估**，不會卡死在某個 RUNNING 葉節點。共享狀態都放在 `bb`（Blackboard）：自身 pose、最近一次看到領導者的世界座標 `last_seen`、領導者 EWMA 速度、行人 TTL 列表、學出來的地圖。

### 3.1 Retreating
領導者距離 < `D_T_MIN`（約 0.8 m）時，沿 `(self → leader)` 反向後退。優先序最高，避免推到領導者。

### 3.2 Following
領導者可見、距離在 `[D_T_MIN, D_T_BASE]` 之間時，目標點放在「領導者軌跡上、與其速度反向 D\_T 公尺處」——也就是踩著領導者走過的腳印，過彎時不會切角穿牆。

### 3.3 Chasing
領導者可見但太遠時用加權 A\* 規劃到 leader 當前世界位置，速度上限隨領導者速度與距離自適應（`v_max = clip(α_v · |v_L| + α_d · dist, V_MIN, V_MAX)`）。

### 3.4 PlannedRecovery（看不見領導者後最關鍵的葉子）
- 目標 = `predicted_leader = last_seen + v_leader × Δt`，但 `Δt > 0.3 s` 後直接退化成 `last_seen`（防止短時間爆衝外推到牆裡）。
- 在每幀 plan\_grid 上用加權 A\* 重算路徑；若當前 plan 的後半段被新看到的牆切到，立即重新規劃。
- 距 `last_seen` < 0.6 m 時回傳 `FAILURE`，把控制權讓給下一個葉節點。

### 3.5 BackupRecovery
PlannedRecovery 失敗、但跟隨者還沒到達 `last_seen` 時，用純朝向控制（`goto_command`）直接朝 `last_seen` 走，繞過規劃模組。這是為了處理 plan\_grid 暫時把 last\_seen 標成 blocked 的邊界情況。

### 3.6 SpiralExpand
已抵達 `last_seen` 仍看不見領導者，採兩階段：
1. 先走到 `predicted_leader` 推算的中心點。
2. 在該點周圍以 0.5 m 半徑為起點向外做環狀搜索，每圈擴大半徑直到再次相機偵測到。

---

## 4. 控制器（diff-drive）

```python
def goto_command(bb, target_xy, v_cap):
    bearing = atan2(ty - by, tx - bx)
    err = wrap_pi(bearing - bb.yaw)
    w = clip(1.6 * err, -1.5, 1.5)
    if abs(err) > 0.6:           # 角度誤差大 → 純旋轉
        return 0.0, w
    v = clip(0.9 * dist, 0, v_cap)
    return v, w
```

**只前進，永不倒車**——避免因為短暫角度誤差倒退離開目標。前方淨距小於 `LIDAR_DANGER_M` 時 `apply_lidar_safety` 會把線速度歸零、原地閃避方向側偏。

---

## 5. 與 Nav2 BT 的對應關係

| Nav2 概念 | 本實作對應 |
| --- | --- |
| `RecoveryNode` 控制流 | `Selector(memory=False)` 自動 fallback |
| `ComputePathToPose` | `astar_weighted` on `plan_grid` |
| `FollowPath` | `Following` / `Chasing` 內的 `goto_command` |
| `Spin / BackUp / Wait` | `BackupRecovery` + `SpiralExpand` |
| Costmap inflation layer | `binary_dilation(struct=disk(INFLATE_RADIUS))` |
| Costmap obstacle layer | `LearnedMap.log_odds` |

差別：Nav2 的 BT 用 BehaviorTree.CPP（XML 描述、tick on demand），本實作用 py\_trees（Python 結構、20 Hz 全樹掃描）。語意一致，後者的 debug 體驗較直接（每個 tick 把整棵樹的狀態 print 出來就能看到誰在跑）。

---

## 6. 評估結果（v34，8 episodes mixed maps，60 s 每集）

```
mean_success:        0.818
mean_loss_ratio:     0.182
collision_count:     0
mean_path_eff:       0.93
follower_stuck_ratio:0.005
crashes:             0
```

加上 10 個 Brownian 行人的 100 s cluttered 場景：success 0.821、0 collisions、path\_eff 0.95。

---

## 7. 移植到 3D 環境的清單

1. `LearnedMap.log_odds` 升級為 voxel grid 或 octree（保留 log-odds 更新邏輯）。
2. `update_from_scan` 改吃 depth camera/pointcloud 而非 2D ranges。
3. `merged_aabbs_from_grid` 直接捨棄，改用 voxel inflation 產 plan\_grid——順便繞過 v34 修掉的 Y-flip 陷阱。
4. A\* 換成 3D（或維持 2.5D，加上 z 限制即可）。
5. 行為樹其餘節點不需改動：所有控制邏輯都建立在 `bb.x / bb.y / bb.yaw / last_seen / leader_vel` 這些感知中性的抽象上。
