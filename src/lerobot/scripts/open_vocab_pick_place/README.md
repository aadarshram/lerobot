# Open Vocabulary Pick-and-Place for LeRobot

Real-world implementation of open vocabulary pick-and-place using vision-language models and LeRobot hardware.

## Features

- **Open Vocabulary Object Detection**: Uses OWL-ViT for zero-shot object detection
- **Natural Language Commands**: Parse instructions using LLMs (Groq/OpenAI)
- **Real Robot Control**: Integrates with LeRobot robots (SO100, Koch, etc.)
- **Camera Calibration**: Pixel-to-world coordinate transformation
- **Modular Design**: Easy to extend and customize

## System Overview

```
Natural Language → Command Parser → Object Detection → Robot Control
     "Pick cup"         ↓                  ↓                 ↓
                   [cup, trc]        Find cup pixels    Execute motion
```

## Installation

### 1. Prerequisites

Make sure you have LeRobot installed and your robot set up. See the [LeRobot installation guide](../../docs/source/installation.mdx).

### 2. Install Additional Dependencies

```bash
pip install transformers pillow opencv-python
```

### 3. LLM API Setup

You need an API key for command parsing. Choose one:

**Option A: Groq (Recommended - Free tier available)**
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Get a free key at: https://console.groq.com/

**Option B: OpenAI**
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

## Quick Start

### Test Without Robot (Mock Mode)

```bash
cd src/lerobot/scripts/open_vocab_pick_place

# Test in mock mode
python main.py --mock --debug --instruction "pick the cup and place in the corner"
```

### Run with Real Robot

```bash
# Interactive mode
python main.py --robot so100 --camera 0

# Single instruction
python main.py --robot so100 --camera 0 --instruction "pick the bottle and place it in the middle"
```

### Command-Line Options

```
--robot       Robot type (so100, koch, etc.) [default: so100]
--camera      Camera device index [default: 0]
--llm         LLM provider (groq/openai) [default: groq]
--calibration Path to calibration file [optional]
--instruction Single instruction to execute [optional]
--mock        Run without real robot [flag]
--debug       Enable debug visualizations [flag]
```

## Usage Examples

### Example 1: Simple Pick-and-Place

```bash
python main.py --instruction "pick the apple and place it in the top right corner"
```

### Example 2: Multiple Commands

```bash
python main.py --instruction "pick the cup and put it in the middle, then move the phone next to the laptop"
```

### Example 3: Interactive Mode

```bash
python main.py

> pick the bottle and place it near the cup
> move the phone to the center
> quit
```

## Camera Calibration

**IMPORTANT**: The default calibration is a rough approximation. For accurate pick-and-place, you must calibrate your camera.

### Quick Calibration (Recommended)

1. Measure your camera height above the table
2. Edit `camera_utils.py` and update:
   ```python
   self.camera_to_robot_transform[:3, 3] = [0.0, 0.0, YOUR_HEIGHT]
   ```

### Precise Calibration (Advanced)

1. Print a checkerboard pattern (9x6 inner corners)
2. Capture 10-20 images from different angles
3. Run calibration:
   ```python
   from camera_utils import CameraCalibration
   
   cal = CameraCalibration()
   cal.calibrate_with_checkerboard(images)
   cal.save_calibration("my_calibration.json")
   ```
4. Use calibration file:
   ```bash
   python main.py --calibration my_calibration.json
   ```

## Workspace Configuration

Edit `robot_controller.py` to match your robot's workspace:

```python
self.workspace = {
    "x": (0.15, 0.45),  # meters from robot base
    "y": (-0.30, 0.30),
    "z": (0.0, 0.30),
}
```

Measure and adjust based on your table setup.

## Module Documentation

### `perception.py`
- Object detection using OWL-ViT
- Zero-shot detection with text queries
- Visualization tools

### `command_parser.py`
- Natural language to robot commands
- Supports Groq and OpenAI APIs
- Handles multiple pick-place pairs

### `robot_controller.py`
- Robot motion control
- Pick-and-place primitives
- Workspace safety checks

### `camera_utils.py`
- Camera interface
- Calibration tools
- Pixel-to-world transformation

### `main.py`
- Main execution script
- Orchestrates all components
- Interactive and batch modes

## Predefined Locations

Use these keywords in commands:

- `top right corner` or `trc`
- `top left corner` or `tlc`
- `bottom right corner` or `brc`
- `bottom left corner` or `blc`
- `middle` or `mid` or `center`

Example: `"place it in the top right corner"`

## Troubleshooting

### Object Detection Issues

**Problem**: Objects not detected reliably

**Solutions**:
- Lower detection threshold: Edit `perception.py` and set `score_threshold=0.05`
- Improve lighting conditions
- Use more specific object names: "red cup" instead of "cup"
- Adjust camera angle for better view

### Robot Motion Issues

**Problem**: Robot doesn't move or moves incorrectly

**Solutions**:
- Check robot connection: `ls /dev/ttyUSB*` or `/dev/ttyACM*`
- Verify robot type matches your hardware
- Test basic robot control first (see LeRobot examples)
- Check workspace bounds match your setup

### Camera Issues

**Problem**: Camera not found

**Solutions**:
- List cameras: `ls /dev/video*`
- Try different camera index: `--camera 1`
- Check camera permissions
- Test camera: `python camera_utils.py`

### LLM Parsing Issues

**Problem**: Commands not parsed correctly

**Solutions**:
- Check API key is set correctly
- Try different provider: `--llm openai`
- Use simpler instructions
- Check API rate limits

## Advanced Usage

### Custom Objects

Train on custom objects by providing specific text queries:

```python
detector = ObjectDetector()
detections = detector.detect(image, ["my custom object", "specific item"])
```

### Custom Motions

Extend `PickPlaceController` for custom behaviors:

```python
class MyController(PickPlaceController):
    def custom_motion(self, target):
        # Your custom implementation
        pass
```

### Integration with Policies

Combine with learned policies for hybrid control:

```python
from lerobot.common.policies.act.modeling_act import ACTPolicy

# Use learned policy for grasping
# Use open vocab for object selection
```

## Performance Tips

1. **Use GPU**: Detection is much faster on GPU
   ```python
   detector = ObjectDetector(device="cuda")
   ```

2. **Cache detections**: Avoid re-detecting static objects

3. **Optimize camera resolution**: Lower resolution = faster processing
   ```python
   camera = CameraInterface(width=320, height=240)
   ```

4. **Batch commands**: Execute multiple actions in one instruction

## Safety

⚠️ **Important Safety Notes**:

- Always test in mock mode first
- Start with slow motions
- Keep emergency stop within reach
- Clear workspace of obstacles
- Monitor robot during execution
- Respect workspace bounds

## Contributing

Improvements welcome! Areas for contribution:

- Better IK solvers for different robots
- More sophisticated trajectory planning
- Support for additional LLM providers
- Improved calibration tools
- Additional robot primitives (push, slide, etc.)

## License

Same as LeRobot (Apache 2.0)

## Citation

If you use this in research, please cite:

```bibtex
@misc{lerobot_open_vocab_pick_place,
  title={Open Vocabulary Pick-and-Place for LeRobot},
  author={LeRobot Community},
  year={2025},
  url={https://github.com/huggingface/lerobot}
}
```

## Acknowledgments

- OWL-ViT by Google Research
- Groq for fast inference
- LeRobot team and community
- Original PyBullet simulation implementation
