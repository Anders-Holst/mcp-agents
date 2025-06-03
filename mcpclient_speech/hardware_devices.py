#!/usr/bin/env python3

import pyaudio

def list_cameras(max_cameras=10):
    """Returns a dictionary mapping device indices to camera information."""
    import cv2
    log_level = cv2.getLogLevel()
    cv2.setLogLevel(0)
    available_cameras = {}
    for index in range(max_cameras):
        capture = cv2.VideoCapture(index)
        if capture is not None and capture.isOpened():
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = capture.get(cv2.CAP_PROP_FPS)
            available_cameras[index] = f"Video: {width:.0f}x{height:.0f} FPS: {fps:.2f} {capture.getBackendName()}"
            capture.release()
    cv2.setLogLevel(log_level)
    return available_cameras


def list_sound_names():
    """
    Returns a list of available sound devices.
    For devices where the name can't be retrieved, the list entry contains ``None`` instead.
    The index of each device's name is the same as its device index when using PyAudio.
    """
    audio = pyaudio.PyAudio()
    result = []
    try:
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            result.append(device_info.get("name"))
    finally:
        audio.terminate()
    return result


def list_working_microphones(sample_rate=0):
    """
    Returns a dictionary mapping device indices to microphone names, for microphones that can be read from.
    """
    audio = pyaudio.PyAudio()
    result = {}
    try:
        for device_index in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(device_index)
            device_name = device_info.get("name")
            device_sample_rate = 0
            # print(f"{device_index}: {device_info}")
            try:
                device_sample_rate = int(device_info["defaultSampleRate"]) if sample_rate == 0 else sample_rate
                pyaudio_stream = audio.open(
                    input_device_index=device_index, channels=1, format=pyaudio.paInt16,
                    rate=device_sample_rate, input=True
                )
                try:
                    _buffer = pyaudio_stream.read(1024)
                    if not pyaudio_stream.is_stopped(): pyaudio_stream.stop_stream()
                finally:
                    pyaudio_stream.close()
            except Exception:
                continue

            result[device_index] = f"{device_name} default sample rate: {device_sample_rate}"
    finally:
        audio.terminate()
    return result

def find_microphone_index(name: str, audio=None, sample_rate=0) -> (int,str):
    """
    Returns a tuple with device index and name of specified microphone or -1 and empty string
    """
    setup_audio: bool = audio is None
    if setup_audio:
        audio = pyaudio.PyAudio()
    try:
        for device_index in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(device_index)
            device_name = device_info.get("name")
            rate = int(device_info["defaultSampleRate"]) if sample_rate == 0 else sample_rate
            if name not in device_name:
                continue
            try:
                pyaudio_stream = audio.open(
                    input_device_index=device_index, channels=1, format=pyaudio.paInt16,
                    rate=rate, input=True
                )
                try:
                    _buffer = pyaudio_stream.read(1024)
                    if not pyaudio_stream.is_stopped(): pyaudio_stream.stop_stream()
                finally:
                    pyaudio_stream.close()
            except Exception:
                continue

            return device_index, device_name
    finally:
        if setup_audio:
            audio.terminate()
    return -1,''


def main(device_type=None, sample_rate=0):
    if device_type == 'cameras':
        cameras = list_cameras()
        print("Available cameras:")
        for index in sorted(cameras.keys()):
            print(f"{index}: {cameras[index]}")
        return

    if device_type == 'microphones':
        working_microphones = list_working_microphones(sample_rate=sample_rate)
        print("Available microphones:")
        for index in sorted(working_microphones.keys()):
            print(f"{index}: {working_microphones[index]}")
        return

    microphones = list_sound_names()
    print("Available sound devices:")
    for index, item in enumerate(microphones):
        print(f"{index}: {item}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=True, description='List sound, microphones or cameras')
    parser.add_argument('-c', '--cameras', action='store_true',
                        default=False, help='List cameras')
    parser.add_argument('-m', '--microphones', action='store_true',
                        default=False, help='List working microphones')
    parser.add_argument('-r', '--rate', type=int,
                        default=0, help='Sample rate')

    args = parser.parse_args()

    main('cameras' if args.cameras else 'microphones' if args.microphones else None, sample_rate=args.rate)
