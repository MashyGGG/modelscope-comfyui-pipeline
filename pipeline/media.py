from pathlib import Path

from .common import checked_path, sha256


def history_file(entry, node_id, root):
    descriptors = entry.get("outputs", {}).get(str(node_id), {}).get("images", [])
    if len(descriptors) != 1 or descriptors[0].get("type") != "output":
        raise RuntimeError(f"Expected one output file from save node {node_id}: {descriptors}")
    item = descriptors[0]
    filename = item["filename"]
    if Path(filename).name != filename or "\\" in filename:
        raise RuntimeError("Unsafe output filename")
    result = checked_path(root, Path(item.get("subfolder", "")) / filename)
    if not result.is_file() or result.stat().st_size == 0:
        raise RuntimeError(f"Missing/empty generated output: {result}")
    return result


def png(path, expected):
    from PIL import Image
    with Image.open(path) as im:
        if im.format != "PNG" or im.size != (expected["width"], expected["height"]):
            raise RuntimeError("Generated PNG has wrong format or resolution")
        im.verify()
    with Image.open(path) as im:
        im.load()
    return {"path": str(path), "sha256": sha256(path), "width": expected["width"], "height": expected["height"]}


def video(path, expected):
    import av
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise RuntimeError("Missing/empty video")
    with av.open(str(path)) as container:
        if "mp4" not in container.format.name or Path(path).suffix.lower() != ".mp4":
            raise RuntimeError("Output is not MP4")
        streams = container.streams.video
        if len(streams) != 1:
            raise RuntimeError("Expected exactly one video stream")
        s = streams[0]
        fps = float(s.average_rate or 0)
        if s.codec_context.name != "h264":
            raise RuntimeError(f"Expected H.264, got {s.codec_context.name}")
        if (s.width, s.height) != (expected["width"], expected["height"]) or abs(fps - expected["fps"]) > 0.001:
            raise RuntimeError("Video resolution/FPS mismatch")
        count, first, previous = 0, None, None
        for frame in container.decode(s):
            if frame.is_corrupt or frame.pts is None:
                raise RuntimeError("Corrupt video frame or missing timestamp")
            t = float(frame.pts * frame.time_base)
            if previous is not None and abs((t - previous) - 1 / fps) > 0.002:
                raise RuntimeError("Nonuniform video timestamps")
            if first is None:
                first = t
            previous = t
            count += 1
        if count != expected["frames"]:
            raise RuntimeError(f"Decoded frame count {count} != expected {expected['frames']}")
        duration = previous - first + 1 / fps
        expected_duration = expected["frames"] / expected["fps"]
        container_duration = container.duration / av.time_base if container.duration is not None else duration
        if abs(duration - expected_duration) > 1 / fps or abs(container_duration - expected_duration) > 1 / fps:
            raise RuntimeError("Video duration mismatch")
    return {"path": str(path), "sha256": sha256(path), "width": expected["width"], "height": expected["height"],
            "fps": fps, "frames": count, "duration_seconds": duration, "codec": "h264", "fully_decoded": True}
