# Vanish Webcam

Disappear from your webcam with a hand gesture — live in Google Meet, Zoom, Teams or any app that lets you pick a camera.

- **Normal** — make a ✌️ V sign and dissolve away; repeat it to fade back in.
- **Naruto** — press your palms together and swap places with a log in a puff of smoke; repeat to return.

Everything runs locally. Video is never recorded or uploaded.

## Setup

Tested on macOS (Apple Silicon). Needs **Python 3.11** — MediaPipe does not support newer versions yet.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The MediaPipe models download automatically on first run.

## Run

```bash
./vanish normal     # V sign mode
./vanish naruto     # palms-together mode
./vanish            # pick from a menu
```

Optional: `echo "alias vanish='$PWD/vanish'" >> ~/.zshrc` to run `vanish normal` from anywhere.

1. **Step out of the frame** for 4 seconds while the background is captured — completely, or you'll be part of it.
2. Come back and **gesture to vanish**. Lower your hand, then **gesture again to reappear**.

The first run asks for camera access for your terminal — allow it.

## Keys

Keys work when the **Webcam - Preview** window is focused. Gestures work even with it in the background.

| Key | Action |
| --- | --- |
| `W` / `Space` | Recapture the background |
| `V` | Vanish |
| `R` | Reappear |
| `F` | Fullscreen |
| `Q` / `Esc` | Quit |

## Use in meetings

The app streams to **OBS Virtual Camera**. One-time setup:

1. Install OBS: `brew install --cask obs`
2. Open OBS → **Start Virtual Camera** → allow the extension in **System Settings → General → Login Items & Extensions → Camera Extensions**. Then stop it and quit OBS.

Each call: start the app, then pick **OBS Virtual Camera** as the camera in Meet/Zoom (restart the browser if it isn't listed). Turn off the meeting app's own background blur.

Your preview is mirrored; the meeting gets the true, unmirrored image so text reads correctly.

## Options

| Flag | Purpose |
| --- | --- |
| `--camera 1` | Use another webcam |
| `--width 1280 --height 720` | Capture resolution |
| `--fps 30` | Max virtual-camera FPS |
| `--no-virtual-camera` | Preview window only |
| `--no-mirror` | Don't mirror the preview |
| `--gesture-debug` | Print gesture detection status |

## Troubleshooting

- **You're still visible after vanishing** — you were in the frame during background capture. Press `W` and step fully out.
- **"OBS Virtual Camera is not installed"** — finish the OBS one-time setup above.
- **Logs** — every run is recorded with timestamps in `logs/vanish-<date>.log`.

## Tests

```bash
.venv/bin/python -m unittest
```

## Credits

Based on [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam). The log sprite is an original generated illustration; smoke is drawn procedurally.
