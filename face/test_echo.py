"""Visual AEC test tool with VU meters and oscilloscope.

Plays TTS through speakers via the full-duplex EchoDetector (WebRTC AEC).
Shows a live OpenCV window with:
- VU meters for output, raw mic, and clean (echo-cancelled) signal
- Oscilloscope waveforms for all three signals
- Barge-in threshold + trigger state

Speak while TTS is playing to test barge-in detection.

Usage:
    pixi run python test_echo.py
    pixi run python test_echo.py --threshold 0.05
    pixi run python test_echo.py --text "Custom text to speak"
"""

import time
import argparse
import logging
import numpy as np
import cv2
import scipy.signal

from voice_input import EchoDetector
from voice_output import VoiceOutput

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

DEFAULT_TEXT = (
    "Hello! This is a long test of the echo cancellation system. "
    "I am going to keep talking for a while so you have time to test "
    "the barge-in detection. Try speaking loudly while I am talking. "
    "If the system works correctly, it should detect your voice and "
    "stop my speech. The WebRTC echo canceller should remove my voice "
    "from the microphone signal, leaving only your voice in the clean "
    "output. Let us see if it works!"
)

# Colors
C_OUTPUT = (0, 200, 200)    # yellow-ish
C_RAW = (0, 180, 0)         # green
C_CLEAN = (255, 180, 0)     # blue-ish
C_CLEAN_HOT = (0, 0, 255)   # red
C_THRESH = (0, 255, 255)    # cyan
C_DIM = (80, 80, 80)
C_TEXT = (200, 200, 200)


def draw_vu_bar(frame, x, y, w, h, level, max_val, color, label):
    """Draw a vertical VU meter bar."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    bar_h = int(min(1.0, level / max(max_val, 1e-6)) * h)
    if bar_h > 0:
        for dy in range(bar_h):
            frac = dy / h
            c = color if frac < 0.7 else (0, 220, 220) if frac < 0.9 else (0, 0, 255)
            cv2.line(frame, (x + 1, y + h - dy), (x + w - 1, y + h - dy), c, 1)

    cv2.rectangle(frame, (x, y), (x + w, y + h), C_DIM, 1)
    cv2.putText(frame, label, (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    cv2.putText(frame, f"{level:.3f}", (x - 4, y + h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_TEXT, 1)


def draw_scope(frame, x, y, w, h, samples, color, label, amplitude=1.0):
    """Draw an oscilloscope waveform."""
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Center line
    mid_y = y + h // 2
    cv2.line(frame, (x, mid_y), (x + w, mid_y), (40, 40, 40), 1)

    # Downsample waveform to screen width
    n = len(samples)
    if n < 2:
        return
    step = max(1, n // w)
    points = []
    for i in range(0, min(n, w * step), step):
        px = x + len(points)
        val = float(samples[i])
        py = int(mid_y - val / max(amplitude, 1e-6) * (h // 2))
        py = max(y, min(y + h, py))
        points.append((px, py))

    # Draw waveform
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 1)

    # Border and label
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_DIM, 1)
    cv2.putText(frame, label, (x + 4, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def run_test(text: str, threshold: float):
    print(f"  Threshold: {threshold}")
    print()

    tts = VoiceOutput()
    print("Loading TTS model...")
    tts.load_sync()
    if not tts.ready:
        print("TTS failed to load")
        return

    echo = EchoDetector(speech_threshold=threshold)

    # Synthesize and feed TTS
    tts_sr = tts._piper_voice.config.sample_rate
    stream_sr = echo._stream_rate

    print("Synthesizing TTS...")
    for chunk in tts._piper_voice.synthesize(text):
        raw = chunk.audio_float_array
        if tts_sr != stream_sr:
            resampled = scipy.signal.resample(
                raw, int(len(raw) * stream_sr / tts_sr))
            echo.feed(resampled.astype(np.float32))
        else:
            echo.feed(raw)
    echo.finish_feeding()

    # Start playback
    echo.start(tts_sample_rate=stream_sr)
    print("Playing. Speak to test barge-in. Press Q to quit.\n")

    win_w, win_h = 900, 700
    cv2.namedWindow("AEC Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AEC Test", win_w, win_h)

    triggered = False
    start_time = time.time()

    while echo.active:
        elapsed = time.time() - start_time
        frame = np.zeros((win_h, win_w, 3), dtype=np.uint8)

        # --- Title ---
        state = "BARGE-IN!" if echo.user_speaking else "Playing TTS..."
        state_color = C_CLEAN_HOT if echo.user_speaking else (0, 255, 0)
        cv2.putText(frame, f"WebRTC AEC Test - {state}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        cv2.putText(frame, f"Time: {elapsed:.1f}s | Threshold: {threshold} | Q=quit",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TEXT, 1)

        # --- VU meters (left column) ---
        vu_x = 20
        vu_y = 70
        vu_w = 40
        vu_h = 180
        vu_gap = 25

        draw_vu_bar(frame, vu_x, vu_y, vu_w, vu_h,
                    echo.output_rms, 1.0, C_OUTPUT, "OUT")
        draw_vu_bar(frame, vu_x + vu_w + vu_gap, vu_y, vu_w, vu_h,
                    echo.current_rms, 1.0, C_RAW, "RAW")

        clean_color = C_CLEAN_HOT if echo.user_speaking else C_CLEAN
        draw_vu_bar(frame, vu_x + 2 * (vu_w + vu_gap), vu_y, vu_w, vu_h,
                    echo.clean_rms, 0.3, clean_color, "CLEAN")

        # Threshold line on clean bar
        cx = vu_x + 2 * (vu_w + vu_gap)
        thresh_frac = min(1.0, threshold / 0.3)
        thresh_y = vu_y + vu_h - int(thresh_frac * vu_h)
        cv2.line(frame, (cx - 5, thresh_y), (cx + vu_w + 5, thresh_y), C_THRESH, 2)

        # Stats text
        sx = vu_x + 3 * (vu_w + vu_gap) + 10
        if echo.current_rms > 0.001:
            reduction = echo.current_rms / max(echo.clean_rms, 0.0001)
            cv2.putText(frame, f"Reduction: {reduction:.0f}x",
                        (sx, vu_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT, 1)
        cv2.putText(frame, f"Out:   {echo.output_rms:.4f}",
                    (sx, vu_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_OUTPUT, 1)
        cv2.putText(frame, f"Raw:   {echo.current_rms:.4f}",
                    (sx, vu_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_RAW, 1)
        cv2.putText(frame, f"Clean: {echo.clean_rms:.4f}",
                    (sx, vu_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, clean_color, 1)

        # --- Oscilloscope (bottom section) ---
        scope_x = 20
        scope_w = win_w - 40
        scope_h = 120
        scope_gap = 10

        scope_y1 = vu_y + vu_h + 50
        draw_scope(frame, scope_x, scope_y1, scope_w, scope_h,
                   echo.wave_output, C_OUTPUT, "Output (speakers)", amplitude=1.0)

        scope_y2 = scope_y1 + scope_h + scope_gap
        draw_scope(frame, scope_x, scope_y2, scope_w, scope_h,
                   echo.wave_raw, C_RAW, "Raw mic (with echo)", amplitude=1.0)

        scope_y3 = scope_y2 + scope_h + scope_gap
        draw_scope(frame, scope_x, scope_y3, scope_w, scope_h,
                   echo.wave_clean, clean_color,
                   "Clean (echo cancelled)", amplitude=0.3)

        # Threshold lines on clean scope
        clean_mid = scope_y3 + scope_h // 2
        t_offset = int(threshold / 0.3 * (scope_h // 2))
        cv2.line(frame, (scope_x, clean_mid - t_offset),
                 (scope_x + scope_w, clean_mid - t_offset), C_THRESH, 1)
        cv2.line(frame, (scope_x, clean_mid + t_offset),
                 (scope_x + scope_w, clean_mid + t_offset), C_THRESH, 1)

        cv2.imshow("AEC Test", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break

        if echo.user_speaking and not triggered:
            triggered = True
            echo.stop(beep=True)
            for _ in range(30):
                cv2.putText(frame, "BARGE-IN!", (300, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, C_CLEAN_HOT, 3)
                cv2.imshow("AEC Test", frame)
                cv2.waitKey(30)
            break

    echo.stop()
    cv2.destroyAllWindows()

    print(f"  Result: {'BARGE-IN' if triggered else 'Finished normally'}")
    print(f"  Duration: {time.time() - start_time:.1f}s")
    if triggered:
        print(f"  Clean RMS at trigger: {echo.clean_rms:.6f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Visual WebRTC AEC test")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to speak")
    parser.add_argument("--threshold", type=float, default=0.08,
                        help="Barge-in threshold (default 0.08)")
    args = parser.parse_args()
    run_test(args.text, args.threshold)


if __name__ == "__main__":
    main()
