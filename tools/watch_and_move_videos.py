"""Watch D:\\Downloads for newly downloaded videos, crop to 640x640, and move to videos/ with timestamp names."""

import argparse
from datetime import datetime
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional, Set, Tuple

import cv2
from watchdog.events import FileCreatedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".flv",
    ".wmv",
    ".webm",
    ".m4v",
    ".ts",
    ".3gp",
    ".mpeg",
    ".mpg",
}

TEMP_EXTENSIONS = {
    ".crdownload",
    ".part",
    ".tmp",
    ".download",
    ".aria2",
    ".filepart",
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("VideoWatcher")


def get_ffmpeg_binary() -> Optional[str]:
    """Find FFmpeg executable via system PATH or imageio_ffmpeg."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def get_video_dimensions(video_path: Path) -> Tuple[int, int]:
    """Get video width and height using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def is_video_file(path: Path) -> bool:
    """Check if the given path has a supported video extension and is not a temp download."""
    if path.name.startswith("."):
        return False
    suffix = path.suffix.lower()
    if suffix in TEMP_EXTENSIONS:
        return False
    for temp_ext in TEMP_EXTENSIONS:
        if path.name.lower().endswith(temp_ext):
            return False
    return suffix in VIDEO_EXTENSIONS


def crop_video_file(
    input_path: Path,
    output_path: Path,
    target_w: int = 640,
    target_h: int = 640,
) -> bool:
    """
    Crop video to target_w x target_h.
    Horizontal alignment: centered ((in_w - target_w) / 2).
    Vertical alignment: bottom (in_h - target_h).
    """
    w, h = get_video_dimensions(input_path)
    if w == 0 or h == 0:
        logger.error("Could not read video dimensions for '%s'", input_path.name)
        return False

    ffmpeg_bin = get_ffmpeg_binary()
    if ffmpeg_bin:
        # Construct filter: scale if smaller than target, then crop centered-x, bottom-y
        filters = []
        if w < target_w or h < target_h:
            scale_factor = max(target_w / w, target_h / h)
            scaled_w = math.ceil(w * scale_factor)
            scaled_h = math.ceil(h * scale_factor)
            # Ensure even dimensions for h264 encoder
            scaled_w = scaled_w if scaled_w % 2 == 0 else scaled_w + 1
            scaled_h = scaled_h if scaled_h % 2 == 0 else scaled_h + 1
            filters.append(f"scale={scaled_w}:{scaled_h}")

        filters.append(f"crop={target_w}:{target_h}:(in_w-{target_w})/2:in_h-{target_h}")
        filter_str = ",".join(filters)

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(input_path),
            "-vf",
            filter_str,
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-map",
            "0:v",
            "-map",
            "0:a?",
            "-c:a",
            "copy",
            str(output_path),
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error("FFmpeg cropping failed for '%s': %s", input_path.name, e.stderr)
            return False
    else:
        # Fallback using OpenCV VideoCapture and VideoWriter
        logger.info("Using OpenCV fallback for video cropping...")
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_w, target_h))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                fh, fw = frame.shape[:2]
                if fw < target_w or fh < target_h:
                    scale = max(target_w / fw, target_h / fh)
                    frame = cv2.resize(frame, (math.ceil(fw * scale), math.ceil(fh * scale)))
                    fh, fw = frame.shape[:2]

                x_start = max(0, (fw - target_w) // 2)
                y_start = max(0, fh - target_h)
                cropped_frame = frame[y_start : y_start + target_h, x_start : x_start + target_w]
                writer.write(cropped_frame)
            return True
        except Exception as e:
            logger.error("OpenCV cropping failed for '%s': %s", input_path.name, e)
            return False
        finally:
            cap.release()
            writer.release()


def wait_for_file_ready(
    file_path: Path, idle_timeout: float = 30.0, poll_interval: float = 1.0
) -> bool:
    """Wait until the file has stopped growing and is unlocked by the download process."""
    if not file_path.exists():
        return False

    last_size = -1
    last_change_time = time.time()

    logger.info("Waiting for '%s' to finish downloading...", file_path.name)

    while True:
        if not file_path.exists():
            logger.warning("File '%s' disappeared during wait.", file_path.name)
            return False

        try:
            current_size = file_path.stat().st_size

            if current_size != last_size:
                last_size = current_size
                last_change_time = time.time()
            else:
                if current_size > 0:
                    try:
                        with open(file_path, "r+b"):
                            return True
                    except (PermissionError, OSError):
                        pass

            if time.time() - last_change_time > idle_timeout:
                try:
                    with open(file_path, "r+b"):
                        return current_size > 0
                except (PermissionError, OSError):
                    logger.error("Timed out waiting for lock release on '%s'.", file_path.name)
                    return False

        except (PermissionError, OSError):
            pass

        time.sleep(poll_interval)


def generate_timestamp_name(dest_dir: Path, ext: str) -> Path:
    """Generate a unique timestamp-based filename in the destination directory."""
    base_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = dest_dir / f"{base_ts}{ext}"
    counter = 1
    while candidate.exists():
        candidate = dest_dir / f"{base_ts}_{counter}{ext}"
        counter += 1
    return candidate


def crop_existing_destination_videos(
    dest_dir: Path, target_w: int = 640, target_h: int = 640, dry_run: bool = False
):
    """Inspect all video files in dest_dir and crop any that are not target_w x target_h."""
    if not dest_dir.exists():
        return

    for item in dest_dir.iterdir():
        if item.is_file() and is_video_file(item) and not item.name.startswith(".tmp_"):
            w, h = get_video_dimensions(item)
            if w == target_w and h == target_h:
                logger.info("Video '%s' is already %dx%d. Skipping crop.", item.name, w, h)
                continue

            logger.info(
                "Cropping existing video in destination: '%s' (%dx%d -> %dx%d, centered-x, bottom-y)...",
                item.name,
                w,
                h,
                target_w,
                target_h,
            )
            if dry_run:
                logger.info("[DRY RUN] Would crop '%s'", item.name)
                continue

            temp_output = dest_dir / f".tmp_{item.name}"
            success = crop_video_file(
                input_path=item,
                output_path=temp_output,
                target_w=target_w,
                target_h=target_h,
            )
            if success and temp_output.exists() and temp_output.stat().st_size > 0:
                # Replace original file with cropped version
                try:
                    item.unlink()
                    temp_output.rename(item)
                    logger.info("Successfully cropped existing video: '%s'", item.name)
                except Exception as e:
                    logger.error("Failed to replace '%s' with cropped version: %s", item.name, e)
                    if temp_output.exists():
                        temp_output.unlink()
            else:
                logger.error("Failed to crop '%s'", item.name)
                if temp_output.exists():
                    temp_output.unlink()


def process_video_file(
    src_path: Path,
    dest_dir: Path,
    target_w: int = 640,
    target_h: int = 640,
    idle_timeout: float = 30.0,
    poll_interval: float = 1.0,
    dry_run: bool = False,
) -> Optional[Path]:
    """Wait for file readiness, crop to target_w x target_h (centered-x, bottom-y), move and rename with timestamp."""
    if not is_video_file(src_path):
        return None

    if not wait_for_file_ready(src_path, idle_timeout=idle_timeout, poll_interval=poll_interval):
        logger.warning("File '%s' was not ready or could not be accessed. Skipping.", src_path.name)
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    target_path = generate_timestamp_name(dest_dir, src_path.suffix.lower())
    temp_target = dest_dir / f".tmp_{target_path.name}"

    if dry_run:
        logger.info(
            "[DRY RUN] Would crop (%dx%d bottom-aligned) and move '%s' -> '%s'",
            target_w,
            target_h,
            src_path,
            target_path,
        )
        return target_path

    logger.info(
        "Cropping and moving '%s' -> '%s' (%dx%d centered-x, bottom-y)...",
        src_path.name,
        target_path.name,
        target_w,
        target_h,
    )

    crop_success = crop_video_file(
        input_path=src_path,
        output_path=temp_target,
        target_w=target_w,
        target_h=target_h,
    )

    if crop_success and temp_target.exists() and temp_target.stat().st_size > 0:
        try:
            temp_target.rename(target_path)
            # Remove original from source downloads directory
            src_path.unlink()
            logger.info("Successfully cropped and moved: '%s' -> '%s'", src_path.name, target_path.name)
            return target_path
        except Exception as e:
            logger.error("Failed to finalize moved video '%s': %s", target_path.name, e)
            if temp_target.exists():
                temp_target.unlink()
            return None
    else:
        logger.warning("Cropping failed for '%s'. Moving original as fallback.", src_path.name)
        if temp_target.exists():
            temp_target.unlink()
        try:
            shutil.move(str(src_path), str(target_path))
            logger.info("Moved uncropped fallback: '%s' -> '%s'", src_path.name, target_path.name)
            return target_path
        except Exception as e:
            logger.error("Failed to move fallback file '%s': %s", src_path.name, e)
            return None


class VideoDownloadHandler(FileSystemEventHandler):
    """Event handler that detects newly created or renamed video files."""

    def __init__(
        self,
        dest_dir: Path,
        target_w: int = 640,
        target_h: int = 640,
        idle_timeout: float = 30.0,
        poll_interval: float = 1.0,
        dry_run: bool = False,
    ):
        super().__init__()
        self.dest_dir = dest_dir
        self.target_w = target_w
        self.target_h = target_h
        self.idle_timeout = idle_timeout
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self.processing_paths: Set[Path] = set()
        self.lock = threading.Lock()

    def _schedule_processing(self, file_path: Path):
        with self.lock:
            if file_path in self.processing_paths:
                return
            self.processing_paths.add(file_path)

        def worker():
            try:
                process_video_file(
                    src_path=file_path,
                    dest_dir=self.dest_dir,
                    target_w=self.target_w,
                    target_h=self.target_h,
                    idle_timeout=self.idle_timeout,
                    poll_interval=self.poll_interval,
                    dry_run=self.dry_run,
                )
            finally:
                with self.lock:
                    self.processing_paths.discard(file_path)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if is_video_file(path):
            logger.info("New video file detected (created): '%s'", path.name)
            self._schedule_processing(path)

    def on_moved(self, event: FileMovedEvent):
        if event.is_directory:
            return
        dest_path = Path(event.dest_path)
        if is_video_file(dest_path):
            logger.info(
                "New video file detected (download completed/renamed): '%s' -> '%s'",
                Path(event.src_path).name,
                dest_path.name,
            )
            self._schedule_processing(dest_path)


def process_existing_source_videos(
    source_dir: Path,
    dest_dir: Path,
    target_w: int = 640,
    target_h: int = 640,
    idle_timeout: float = 30.0,
    poll_interval: float = 1.0,
    dry_run: bool = False,
):
    """Scan and process any existing video files currently in source_dir."""
    logger.info("Checking for existing videos in source '%s'...", source_dir)
    for item in source_dir.iterdir():
        if item.is_file() and is_video_file(item):
            logger.info("Found existing video in source: '%s'", item.name)
            process_video_file(
                src_path=item,
                dest_dir=dest_dir,
                target_w=target_w,
                target_h=target_h,
                idle_timeout=idle_timeout,
                poll_interval=poll_interval,
                dry_run=dry_run,
            )


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_source = Path(r"D:\Downloads")
    default_dest = repo_root / "videos"

    parser = argparse.ArgumentParser(
        description="Watch a directory for newly downloaded videos, crop to 640x640 (centered horizontally, bottom vertically), and move with timestamp names."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source,
        help=f"Directory to watch for downloads (default: {default_source})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=default_dest,
        help=f"Destination directory for videos (default: {default_dest})",
    )
    parser.add_argument(
        "--crop-width",
        type=int,
        default=640,
        help="Target crop width in pixels (default: 640)",
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=640,
        help="Target crop height in pixels (default: 640)",
    )
    parser.add_argument(
        "--crop-dest-existing",
        action="store_true",
        default=True,
        help="Crop existing videos in the destination folder if they are not already 640x640 (default: True).",
    )
    parser.add_argument(
        "--no-crop-dest-existing",
        dest="crop_dest_existing",
        action="store_false",
        help="Do not crop existing videos in destination folder.",
    )
    parser.add_argument(
        "--process-source-existing",
        action="store_true",
        default=True,
        help="Process matching video files already in the source downloads folder before watching (default: True).",
    )
    parser.add_argument(
        "--no-process-source-existing",
        dest="process_source_existing",
        action="store_false",
        help="Do not process existing video files in source downloads folder.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Watch subdirectories recursively (default: False).",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=30.0,
        help="Seconds of stable file size before considering download finished (default: 30.0).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without moving or cropping files.",
    )

    args = parser.parse_args()

    source_dir = args.source.resolve()
    dest_dir = args.dest.resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        logger.error("Source directory '%s' does not exist or is not a directory.", source_dir)
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Source directory:      %s", source_dir)
    logger.info("Destination directory: %s", dest_dir)
    logger.info("Crop target:           %dx%d (centered horizontally, bottom vertically)", args.crop_width, args.crop_height)
    logger.info("Supported formats:     %s", ", ".join(sorted(VIDEO_EXTENSIONS)))
    logger.info("Recursive watching:    %s", args.recursive)

    # 1. Crop any existing videos in the destination directory that aren't already 640x640
    if args.crop_dest_existing:
        crop_existing_destination_videos(
            dest_dir=dest_dir,
            target_w=args.crop_width,
            target_h=args.crop_height,
            dry_run=args.dry_run,
        )

    # 2. Process any videos currently waiting in source_dir (D:\Downloads)
    if args.process_source_existing:
        process_existing_source_videos(
            source_dir=source_dir,
            dest_dir=dest_dir,
            target_w=args.crop_width,
            target_h=args.crop_height,
            idle_timeout=args.idle_timeout,
            poll_interval=args.poll_interval,
            dry_run=args.dry_run,
        )

    # 3. Watch for new downloads
    handler = VideoDownloadHandler(
        dest_dir=dest_dir,
        target_w=args.crop_width,
        target_h=args.crop_height,
        idle_timeout=args.idle_timeout,
        poll_interval=args.poll_interval,
        dry_run=args.dry_run,
    )

    observer = Observer()
    observer.schedule(handler, str(source_dir), recursive=args.recursive)
    observer.start()

    logger.info("Watching for new video downloads... Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping watcher...")
        observer.stop()
    observer.join()
    logger.info("Watcher stopped.")


if __name__ == "__main__":
    main()
