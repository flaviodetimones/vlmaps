# Architecture Analysis — HSSD Integration Planning

Generated: 2026-03-30
Branch: hssd-integration

---

## 1. Directory Overview

```
third_party/vlmaps/
├── application/
│   ├── interactive_object_nav.py   # Main nav pipeline (LLM + VLMap + YOLOE)
│   ├── create_map.py               # VLMap construction orchestrator
│   ├── generate_obstacle_map_png.py # Top-down map export for LabelMe
│   ├── labelme_to_room_map.py      # LabelMe JSON → room_map grid
│   ├── place_objects.py            # Custom object placement via JSON config
│   ├── yoloe_worker.py             # YOLOE subprocess (avoids CUDA/OpenGL conflict)
│   └── dataset/collect_custom_dataset.py  # RGB-D + pose collection from Habitat
├── vlmaps/
│   ├── robot/habitat_lang_robot.py # HabitatLanguageRobot: sim init, visgraph, nav
│   ├── utils/
│   │   ├── habitat_utils.py        # make_cfg(): Habitat-Sim configuration builder
│   │   ├── matterport3d_categories.py  # MP3D 42-category vocabulary (mp3dcat)
│   │   ├── llm_utils.py            # LLM mapper: object → VLMap categories + rooms
│   │   ├── room_map_utils.py       # Room labels: MP3D vocabulary, load/save/query
│   │   └── habitat_utils.py        # Sensor config, semantic scene access
│   └── map/vlmap.py                # VLMap: 3D voxel map, CLIP feature storage
├── config/
│   ├── data_paths/
│   │   ├── docker.yaml             # /workspace/data/{mp3d,vlmaps_dataset}
│   │   ├── default.yaml            # /home/mario/tfg/data/hm3d
│   │   └── hssd.yaml               # /workspace/data/{hssd-hab,vlmaps_dataset_hssd}
│   ├── map_config/vlmaps.yaml      # Grid size, cell size, categories
│   ├── params/default.yaml         # turn_angle=5, forward=0.1, sensor settings
│   └── object_goal_navigation_cfg.yaml  # Top-level nav config (dataset_type added)
└── docker/menu.sh                  # Interactive Docker menu
```

---

## 2. Module Roles and MP3D Dependencies

### scene-agnostic (no changes needed)

| Module | Role | Status |
|--------|------|--------|
| `vlmap.py` | 3D voxel feature accumulation, CLIP scoring | ✅ scene-agnostic |
| `interactive_object_nav.py` (nav logic) | BFS, visgraph, collision recovery, visual centering | ✅ scene-agnostic |
| `yoloe_worker.py` | YOLOE detection subprocess | ✅ scene-agnostic |
| `place_objects.py` | Custom object placement | ✅ scene-agnostic |
| `labelme_to_room_map.py` | LabelMe → room grid | ✅ scene-agnostic |
| `llm_utils.py` (parsing) | Instruction parsing, room reasoning | ✅ scene-agnostic |
| `room_map_utils.py` (I/O) | load/save/find_room_goal | ✅ scene-agnostic |

### MP3D-specific (changes needed for HSSD)

| File | Lines | Dependency | Migration strategy |
|------|-------|-----------|-------------------|
| `matterport3d_categories.py` | 1-42 | `mp3dcat` list (42 categories) | Add `get_categories(dataset_type)` helper |
| `interactive_object_nav.py` | 36, 1388, 1418 | `from matterport3d_categories import mp3dcat` | Use `get_categories()` |
| `llm_utils.py` | 154, 169, 176 | `mp3dcat` import + usage | Pass category list as parameter |
| `habitat_lang_robot.py` | 110, 122-134 | `mp3dcat.copy()` + MP3D/HM3D .glb naming | Add HSSD branch |
| `habitat_utils.py` | 16 | `sim_cfg.scene_id = settings["scene"]` | Add `scene_dataset_config_file` support |
| `room_map_utils.py` | 25-50, 151 | `MP3D_REGION_LETTER_MAP`, `parse_house_file()` | Unused for HSSD (LabelMe path) |
| `config/data_paths/docker.yaml` | 1 | `habitat_scene_dir: /workspace/data/mp3d` | Added `hssd.yaml` variant |
| `config/map_config/vlmaps.yaml` | 35 | `categories: "mp3d"` | Added `"hssd"` option |
| `docker/menu.sh` | 130 | `HABITAT_DIR=/workspace/data/mp3d` | Added dataset_type selector |

---

## 3. VLMap Data Directory Structure

```
vlmaps_dataset/<SceneName>_<id>/
├── rgb/              000000.png, 000001.png ...   (captured frames)
├── depth/            000000.npy, ...              (depth arrays)
├── semantic/         000000.npy, ...              (semantic label arrays)
├── poses.txt         N×7 (x, y, z, qx, qy, qz, qw)
├── vlmap/
│   └── vlmaps.h5df   H5: grid_feat, grid_pos, weight, grid_rgb, occupied_ids
├── room_map/         (optional, added by LabelMe workflow)
│   ├── room_map.npy
│   ├── regions.json
│   └── room_map_viz.png
├── obstacle_map.png
├── topdown_rgb.png
└── topdown_labeled.png
```

This structure is **fully dataset-agnostic**. HSSD scenes will use the same layout.

---

## 4. Scene Loading Flow

### Current (MP3D / HM3D)

```
config.scene_id (int)
  → vlmaps_data_save_dirs[scene_id]          # e.g. data/vlmaps_dataset/mJXqzFtmKg4_1
  → scene_name = dir.name.split("_")[0]       # "mJXqzFtmKg4"
  → _setup_sim(scene_name)
      → if "-" in name: glb = name_short + ".basis.glb"
        else:           glb = name + ".glb"
      → sim_cfg.scene_id = /path/to/mp3d/mJXqzFtmKg4/mJXqzFtmKg4.glb
      → habitat_sim.Simulator(cfg)
```

### Required for HSSD

```
config.scene_id (int)
  → vlmaps_data_save_dirs[scene_id]          # e.g. data/vlmaps_dataset_hssd/102344280_1
  → scene_name = dir.name.split("_")[0]       # "102344280"
  → _setup_sim(scene_name)
      → if dataset_type == "hssd":
            sim_cfg.scene_id = "102344280"    # just the ID
            sim_cfg.scene_dataset_config_file = /path/to/hssd-hab/hssd-hab.scene_dataset_config.json
        else: [existing MP3D/HM3D logic]
      → habitat_sim.Simulator(cfg)
```

**Key difference**: HSSD requires `scene_dataset_config_file` and uses a bare scene ID (not a path).

---

## 5. Category System

### Current
- `mp3dcat`: 42 categories (chair, table, sofa, bed, sink, toilet, ...) from Matterport3D
- Used to initialize VLMap CLIP scoring and LLM prompts
- "Direct" categories: in mp3dcat → VLMap confirms, no YOLOE needed
- "Indirect" categories: not in mp3dcat → LLM maps to mp3dcat → YOLOE verifies

### HSSD Plan
- Add `get_categories(dataset_type)` in `matterport3d_categories.py`
- For `"hssd"`: return a common furniture list (chair, table, sofa, bed, counter, shelving, etc.)
- CLIP works with any text labels — no retraining needed
- For HSSD, all queries go through LLM mapping (no "direct" shortcut since HSSD has 466 categories)

---

## 6. Room Labeling

| Dataset | Room source | Method |
|---------|------------|--------|
| MP3D | .house file OR LabelMe | `parse_house_file()` or `labelme_to_room_map.py` |
| HSSD | LabelMe (Phase 1) OR `sim.semantic_scene` API (future) | `labelme_to_room_map.py` |

For Phase 2 (HSSD), LabelMe workflow is used — identical to MP3D. The semantic scene API approach (automatic room labels from HSSD annotations) is implemented in `room_provider.py` as `SemanticSceneRoomProvider` for future use.

---

## 7. What Changes in Phase 2

1. **`habitat_utils.py`** — `make_cfg()` accepts optional `scene_dataset_config_file`
2. **`habitat_lang_robot.py`** — `_setup_sim()` adds HSSD branch
3. **`matterport3d_categories.py`** — add `get_categories(dataset_type)` helper
4. **`config/data_paths/hssd.yaml`** — new file pointing to HSSD data paths
5. **`config/object_goal_navigation_cfg.yaml`** — add `dataset_type` and `scene_dataset_config_file` fields
6. **`vlmaps/utils/scene_config.py`** — SceneConfig dataclass
7. **`vlmaps/utils/room_provider.py`** — RoomProvider abstraction (LabelMe + SemanticScene)
