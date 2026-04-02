import subprocess
from pathlib import Path

INPUT_DIR = Path("data/clips/penalties_720p")
OUTPUT_DIR = Path("data/clips/penalties_10s")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

clips = list(INPUT_DIR.glob("*.mp4"))

print(f"Found {len(clips)} clips")

for clip in clips:
    
    output_path = OUTPUT_DIR / clip.name
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(clip),
        "-ss", "5",
        "-t", "10",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] {clip.name}")
    except subprocess.CalledProcessError:
        print(f"[FAIL] {clip.name}")

print("\nDONE")