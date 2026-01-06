# Generate MetaWorld Datasets

This script generates MetaWorld datasets using expert policies and saves them in LeRobot format.

## Prerequisites

```bash
pip install metaworld gymnasium
huggingface-cli login
```

## Usage

### Generate a Single Task Dataset

```bash
python src/lerobot/scripts/generate_datasets.py \
    --task reach-v2 \
    --repo-id username/metaworld-reach-v2 \
    --num-episodes 50
```

### Generate MT10 Benchmark (10 tasks)

```bash
python src/lerobot/scripts/generate_datasets.py \
    --benchmark MT10 \
    --repo-id username/metaworld-mt10 \
    --num-episodes 50
```

This will create 10 separate datasets:
- `username/metaworld-mt10-reach-v2`
- `username/metaworld-mt10-push-v2`
- `username/metaworld-mt10-pick-place-v2`
- etc.

### Generate MT50 Benchmark (50 tasks)

```bash
python src/lerobot/scripts/generate_datasets.py \
    --benchmark MT50 \
    --repo-id username/metaworld-mt50 \
    --num-episodes 50 \
    --push-to-hub
```

This will create 50 separate datasets and upload them to HuggingFace.

## Arguments

### Required Arguments

- `--task TASK` or `--benchmark {MT10,MT50}`: Either specify a single task or a benchmark
- `--repo-id REPO_ID`: HuggingFace repository ID (e.g., `username/metaworld-reach-v2`)

### Dataset Configuration

- `--num-episodes N`: Number of episodes to generate (default: 50)
- `--fps FPS`: Frames per second (default: 80)
- `--root PATH`: Root directory for dataset storage

### Environment Configuration

- `--camera-name {corner,corner2,corner3,topview,behindGripper}`: Camera view (default: corner2)
- `--image-width WIDTH`: Image width in pixels (default: 480)
- `--image-height HEIGHT`: Image height in pixels (default: 480)

### Upload Configuration

- `--push-to-hub`: Upload datasets to HuggingFace Hub
- `--private`: Make datasets private on the Hub

## Available Tasks

### MT10 Tasks (10 tasks)
- reach-v2
- push-v2
- pick-place-v2
- door-open-v2
- drawer-open-v2
- drawer-close-v2
- button-press-topdown-v2
- peg-insert-side-v2
- window-open-v2
- window-close-v2

### MT50 Tasks (50 tasks)
All MT10 tasks plus 40 additional tasks including:
- assembly-v2, basketball-v2, bin-picking-v2, box-close-v2
- coffee-button-v2, coffee-pull-v2, coffee-push-v2, dial-turn-v2
- door-close-v2, door-lock-v2, door-unlock-v2, disassemble-v2
- faucet-close-v2, faucet-open-v2, hammer-v2, hand-insert-v2
- And 24 more...

See the script for the complete list.

## Examples

### 1. Generate a small test dataset locally

```bash
python src/lerobot/scripts/generate_datasets.py \
    --task reach-v2 \
    --repo-id test/metaworld-reach \
    --num-episodes 10
```

### 2. Generate and upload a single task

```bash
python src/lerobot/scripts/generate_datasets.py \
    --task pick-place-v2 \
    --repo-id aadarshram/metaworld-pick-place-v2 \
    --num-episodes 100 \
    --push-to-hub
```

### 3. Generate MT10 with custom camera

```bash
python src/lerobot/scripts/generate_datasets.py \
    --benchmark MT10 \
    --repo-id aadarshram/metaworld-mt10 \
    --num-episodes 50 \
    --camera-name topview \
    --push-to-hub
```

### 4. Generate MT50 in batches (to save disk space)

Generate tasks one at a time by running the script multiple times with different `--task` arguments.

## Dataset Format

The generated datasets follow the LeRobot v3.0 format:

- **Features**:
  - `observation.image`: RGB image (480x480x3 uint8)
  - `observation.state`: Robot state vector (39 float32)
  - `action`: Expert action (4 float32)
  - `next.reward`: Reward signal
  - `next.success`: Success flag
  - `next.done`: Episode termination flag

- **Structure**:
  - Metadata: `meta/tasks.parquet`, `meta/episodes/`
  - Data: `data/chunk-*/file-*.parquet`
  - Videos: `videos/observation.image/chunk-*/file-*.mp4`

## Disk Space Requirements

### Per Task
- **Without videos**: ~50-150 MB per task (50 episodes)
- **With videos**: ~200-500 MB per task (50 episodes)

### Full Benchmarks
- **MT10**: ~1-5 GB (depending on video encoding)
- **MT50**: ~5-25 GB (depending on video encoding)

## Notes

- Expert policies are provided by MetaWorld and achieve near-optimal performance
- Each episode runs for a maximum of 500 steps or until task completion
- Success rate varies by task (typically 60-90%)
- Images are collected at 80 FPS by default
- The `corner2` camera view is recommended for best visual coverage
