# RTC Inference

Real-Time Chunking (RTC) inference scripts for running action chunking policies on physical robots using [HuggingFace LeRobot](https://github.com/huggingface/lerobot).

## Overview

RTC decouples action *prediction* from action *execution* using a two-thread architecture:

- **GetActions thread** — continuously queries the policy for action chunks based on live robot observations, tracking inference latency to stay ahead of the executor.
- **Actor thread** — pulls actions from the queue and sends them to the robot at a fixed frequency (e.g., 30 Hz), ensuring smooth real-time control.

This lets the model keep predicting in the background without blocking the control loop, enabling low-latency robot operation even with slower inference hardware.

## Files

### `eval_with_real_robot.py`

Core RTC inference script with interactive task switching.

**Features:**
- Two-thread RTC architecture (inference + execution)
- Thread-safe robot access via mutex
- Keyboard shortcuts (`1` / `2` / `3`) to switch tasks mid-operation
- Latency compensation — adjusts skip delay for the next chunk based on measured inference time
- Optional `torch.compile` for faster inference (PyTorch 2.0+)
- PEFT / LoRA model support

**Supported policies:** SmolVLA, Pi0, Pi0.5
**Supported robots:** SO100 follower, KOCH follower

### `eval_with_real_robot_entropy.py`

Extended version with entropy-based anomaly detection and automatic recovery.

**Additional features (on top of the base script):**
- **Entropy monitoring** — computes per-dimension action entropy over a rolling 50-action window using histogram-based estimation
- **Automatic recovery** — if entropy exceeds `1.9` for 2+ seconds, the robot switches to a safe recovery task (task 3) for 5 seconds before resuming
- **Rerun visualization** — streams camera feeds and joint states to a [Rerun](https://www.rerun.io/) viewer in real time (local or remote)
- Default execution rate bumped to **30 Hz**

## Quickstart

### Requirements

- [LeRobot](https://github.com/huggingface/lerobot) installed with robot + camera extras
- PyTorch with CUDA or MPS support
- A supported robot (SO100 or KOCH) connected via serial

### Basic usage

```python
from eval_with_real_robot import RTCDemoConfig, demo_cli
import draccus

@draccus.wrap()
def main(cfg: RTCDemoConfig):
    demo_cli(cfg)

main()
```

Or run directly with CLI overrides:

```bash
python eval_with_real_robot.py \
  --policy.path=helper2424/smolvla_check_rtc_last3 \
  --policy.device=cuda \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyUSB0 \
  --task="pick up the red block" \
  --fps=30 \
  --duration=60
```

For entropy-based recovery:

```bash
python eval_with_real_robot_entropy.py \
  --policy.path=helper2424/smolvla_check_rtc_last3 \
  --policy.device=cuda \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyUSB0 \
  --task="pick up the red block" \
  --visualize=true \
  --rerun_ip=127.0.0.1
```

## Configuration

### Policy

| Parameter | Description | Default |
|---|---|---|
| `policy.path` | HuggingFace model path or local checkpoint | — |
| `policy.device` | `cuda`, `mps`, or `cpu` | — |

### Robot

| Parameter | Description |
|---|---|
| `robot.type` | `so100_follower` or `koch_follower` |
| `robot.port` | Serial port (e.g., `/dev/ttyUSB0`) |
| `robot.cameras` | Camera configs (resolution, FPS) |

### RTC

| Parameter | Description | Default |
|---|---|---|
| `rtc.enabled` | Enable Real-Time Chunking | `true` |
| `rtc.execution_horizon` | Actions per chunk | `20` |
| `rtc.max_guidance_weight` | Guidance strength | `1.0` |

### Demo

| Parameter | Description | Default |
|---|---|---|
| `fps` | Action execution frequency | `10` (base), `30` (entropy) |
| `duration` | Run time in seconds | — |
| `task` | Natural language task description | — |
| `action_queue_size_to_get_new_actions` | Queue depth threshold for triggering new predictions | `30` |
| `use_torch_compile` | Enable `torch.compile` for faster inference | `false` |

### Entropy version only

| Parameter | Description | Default |
|---|---|---|
| `visualize` | Enable Rerun streaming | `false` |
| `rerun_session_name` | Rerun session ID | `"rtc_demo"` |
| `rerun_ip` / `rerun_port` | Remote Rerun server address | `127.0.0.1:9876` |
| `compress_images` | JPEG-compress frames before streaming | `false` |

## Architecture

```
Main Thread
├── GetActions Thread
│   ├── Reads latest robot observation
│   ├── Calls policy.predict_action_chunk()
│   ├── Measures inference latency
│   └── Pushes actions → ActionQueue
│
└── Actor Thread
    ├── Pulls actions from ActionQueue
    ├── Sends commands to robot at fixed FPS
    ├── [entropy] Monitors rolling action entropy
    ├── [entropy] Triggers recovery task on high entropy
    └── [entropy] Logs cameras + joints to Rerun
```

## License

Apache 2.0 — Copyright 2025 The HuggingFace Inc. team.
