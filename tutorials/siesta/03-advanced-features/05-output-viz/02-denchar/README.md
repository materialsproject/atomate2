# Tutorial 11: Denchar Visualization Configuration

**Level**: Advanced
**Dataclass Module**: `Denchar` (ADVANCED tier)

## Overview

This tutorial demonstrates how to configure the denchar post-processing utility for generating charge density plots and other grid-based visualizations.

## Available Denchar Parameters

| SIESTA FDF Parameter | Description | Default |
|---------------------|-------------|---------|
| `Write.Denchar` | Enable denchar file output | False |
| `Denchar.NumberPointsX` | Grid points in X direction | 50 |
| `Denchar.NumberPointsY` | Grid points in Y direction | 50 |
| `Denchar.NumberPointsZ` | Grid points in Z direction | 50 |
| `Denchar.XMin` | Minimum X coordinate (Bohr) | Auto |
| `Denchar.XMax` | Maximum X coordinate | Auto |
| `Denchar.YMin` | Minimum Y coordinate | Auto |
| `Denchar.YMax` | Maximum Y coordinate | Auto |
| `Denchar.ZMin` | Minimum Z coordinate | Auto |
| `Denchar.ZMax` | Maximum Z coordinate | Auto |

## Tutorial Example

```bash
cd tutorials/03-advanced-features/05-output-viz/02-denchar
python3 01_basic_denchar.py
```

## Output Files

SIESTA creates:
- `systemLabel.PLD`: Binary grid data
- `systemLabel.DIM`: Grid dimensions metadata

Use `denchar` utility to process these files into XSF/CUBE formats.
