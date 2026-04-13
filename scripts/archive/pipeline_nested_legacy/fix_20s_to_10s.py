from pathlib import Path
import subprocess

INPUT_DIR = Path("data/clips/penalties_720p")
OUTPUT_DIR = Path("data/clips/penalties_10s")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for clip in INPUT_DIR.glob("*.mp4"):
    out_path = OUTPUT_DIR / clip.name

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-y",
        "-ss", "5",
        "-i", str(clip),
        "-t", "10",
        "-c", "copy",
        str(out_path)
    ]

    print("Processing:", clip.name)
    subprocess.run(cmd, check=True)

print("Done.")