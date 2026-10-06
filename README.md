# Rewind Frame

Make a frame with your thumbs and index fingers. Inside it, you see yourself from a few seconds ago. Outside it, everything stays live.

No green screen, no segmentation model, no training. Hand tracking finds your fingertips, and the rest is geometry and pixel blending.

<!-- Add a GIF here. A 5-second clip of waving your hand, then opening the frame and watching the wave replay, sells this better than any paragraph. -->
![demo](demo.gif)

## How it works

1. **Rolling buffer.** Every frame goes into a queue with a timestamp. Frames older than your chosen delay get dropped, so the oldest frame left is always about `delay` seconds old. It uses real timestamps instead of a fixed frame count, so 3 seconds stays 3 seconds even if your webcam runs at 24 fps one minute and 30 the next.
2. **Hand tracking.** MediaPipe Hands finds up to two hands. Out of the 21 landmarks per hand, only two are used: thumb tip and index fingertip.
3. **Smoothing.** Each point goes through an exponential moving average so the frame doesn't shake. Hands are identified by their position on screen (left to right), not by MediaPipe's Left/Right label, which can flip when hands cross.
4. **Shape.** The four points go through `cv2.convexHull`, which always gives a clean outline with no self-intersections.
5. **Compositing.** The hull is filled into a mask, blurred for a soft edge, and used to alpha-blend the old frame over the live one. Only the bounding box of the shape is processed, not the whole image.

## Install

```bash
pip install opencv-python mediapipe numpy
```

MediaPipe publishes wheels for a limited range of Python versions. Python 3.9 to 3.12 is the safe range.

This script uses the older `mp.solutions.hands` API. If you get an error about `solutions` on a newer MediaPipe release, install an older 0.10.x version:

```bash
pip install "mediapipe==0.10.14"
```

## Run

```bash
python rewind_frame.py
```

Click on the video window once so it has keyboard focus. Then hold up both hands and make a rectangle with your thumbs and index fingers. For the first few seconds you'll see "buffering past..." while the delay fills up.

## Controls

| Key | What it does |
|-----|--------------|
| `[` / `]` | Shorter / longer rewind delay (0.5 s steps, 0.5 to 6 s) |
| `-` / `=` | Less / more edge feather |
| `,` / `.` | Less / more smoothing |
| `d` | Show fingertip points and the frame outline |
| `q` | Quit |

## Tuning tips

- **The frame trails behind my hands.** Lower the smoothing with `,`. Less smoothing means more jitter, so find a middle spot.
- **The edge looks hard.** Raise the feather with `=`.
- **Seeing a seam or color shift.** Webcams adjust exposure on their own. If your camera supports it, turn auto-exposure and auto white balance off in your camera software.
- **Using a lot of memory.** The buffer holds raw frames. Lower `FRAME_W` / `FRAME_H` or `MAX_DELAY` at the top of the file.

## Known limits

- It needs both hands. With one hand there are only two points, which can't make a shape.
- Fast hand movement will show a little lag in the mask compared to the live video.
- The delay buffer starts empty, so the effect isn't available until it has filled. The same happens after you increase the delay.
- Resolution is set to 960x720 as a request. Some cameras will give you something else. The code reads the real frame size, so it still works.
- If the camera doesn't open, try changing `cv2.VideoCapture(0)` to `1` or `2`.




https://github.com/user-attachments/assets/e5238e27-7068-45d9-9182-d2986207587c


