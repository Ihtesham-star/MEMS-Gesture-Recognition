# Quick Start Guide

## Basic Setup

### 1. Hardware Connection
- Connect OAK-D Pro camera to USB port
- Connect MEMS mirror system to USB port
- Ensure both devices are recognized by your system

### 2. Software Installation

**Python (on your computer):**
```bash
pip install -r requirements.txt --break-system-packages
```

**MATLAB:**
- Ensure MTI Device SDK is installed
- Add SDK to MATLAB path

### 3. Running the System

**Step 1: Start MATLAB Server (Terminal/Command Window 1)**
```matlab
cd matlab/
fing_tip  % or type: startTcpServer()
```

Wait for message: "Server running on port 30001"

**Step 2: Start Python Client (Terminal 2)**
```bash
cd python/core/
python fing_tip.py
```

## Hand Gestures

| Gesture | Action |
|---------|--------|
| ✊ **Closed Fist** | Lock laser at current position |
| ✋ **Open Hand** | Unlock position and reset |
| 🤏 **Pinch** (thumb+index) | Draw square with scanning pattern |
| 👍 **Thumbs Up** | Toggle locking feature on/off |

## Running Experiments

### Accuracy Comparison (Control vs Proposed Method)

**Without Magnetic Attraction:**
```bash
cd python/experiments/
python NewVirtual.py
```

**With Magnetic Attraction (Your Innovation):**
```bash
cd python/experiments/
python Newway.py
```

Press keys 1-6 to change patterns, 'G' to enable guided mode.

### Game Demo
```bash
cd python/demo/
# First start matlab/nano.m in MATLAB
python nano.py
```

Controls:
- Arrow keys / WASD: Move laser
- G: Grab particle
- R: Release particle
- ESC: Exit

## Troubleshooting

**Problem: "No MEMS devices found"**
- Check USB connection
- Verify MTI Device SDK is installed
- Check device permissions

**Problem: "Connection refused on port 30001"**
- Ensure MATLAB server is running first
- Check if port 30001 is available (not blocked by firewall)

**Problem: "No OAK-D camera detected"**
- Check USB connection
- Try different USB port (USB 3.0 preferred)
- Install DepthAI udev rules (Linux)

**Problem: Hand detection not working**
- Ensure good lighting
- Hand should be visible to camera
- Adjust `min_detection_confidence` parameter if needed

## File Descriptions

### Core System
- `fing_tip.py` + `fing_tip.m`: Main tracking system with fingertip targeting
- `fing_cont.py` + `fintcont.m`: Continuous tracking with square drawing
- `fixed_tip.py`: Stable version with hand orientation detection

### Experiments
- `NewVirtual.py`: Control experiment (standard smoothing only)
- `Newway.py`: Proposed method (with magnetic path attraction)

### Demo
- `nano.py` + `nano.m`: 3D magnetic particle tweezers game

## Data Collection

All experiments automatically save timing data:
- Python: `tracking1_analysis.csv`
- MATLAB: `laser1_response.csv`

Use these files for accuracy analysis and performance metrics.

## Tips for Best Results

1. **Lighting**: Ensure consistent, bright lighting
2. **Hand Position**: Keep hand at comfortable distance (30-60cm from camera)
3. **Movement**: Move smoothly for better tracking
4. **Calibration**: Run system a few seconds before starting gestures
5. **Background**: Plain background works best

## Safety Notes

⚠️ **LASER SAFETY**
- Never look directly into the laser beam
- Ensure laser is pointed at safe surface
- Use appropriate laser safety goggles if needed
- Follow your institution's laser safety protocols

## Support

For issues or questions:
- Email: ihteshamul.hayat@nu.edu.kz
- GitHub Issues: https://github.com/ihtesham-star/MEMS-Gesture-Recognition/issues
