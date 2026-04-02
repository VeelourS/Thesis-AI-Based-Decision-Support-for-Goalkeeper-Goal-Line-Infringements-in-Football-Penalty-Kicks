import cv2
import pandas as pd
from pathlib import Path

# ---------- KONFIGURACJA ----------
CSV_PATH = Path("data/meta/kick_windows_720p.csv")
OUT_DIR = Path("data/yolo/images/train")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Nowy plik CSV, który zmapuje Twoje zdjęcia i momenty kopnięcia
META_OUT = Path("data/yolo/frames_metadata.csv")

# 2 = wyciaga co 2 klatke, zeby wygenerowac duzo danych dla modelu
EVERY_N = 2 
# ----------------------------------

def main():
    if not CSV_PATH.exists():
        print(f"BŁĄD: Brak pliku {CSV_PATH}. Najpierw zrob Krok 1!")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Zaczynam robic zdjecia z {len(df)} krotkich klipow...")

    frames_data = []

    for _, row in df.iterrows():
        video_path = Path(row['window_file'])
        if not video_path.exists():
            continue

        # Pobieramy czas kopnięcia wewnątrz 4-sekundowego okna (zazwyczaj 1.5s)
        kick_in_window_s = float(row['kick_in_window_s'])

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        # Obliczamy, która to dokładnie klatka w tym małym filmiku
        kick_frame_idx = int(round(kick_in_window_s * fps))

        count = 0
        saved = 0
        
        while True:
            ok, frame = cap.read()
            if not ok: break
            
            if count % EVERY_N == 0:
                img_name = f"{video_path.stem}_{saved:04d}.jpg"
                cv2.imwrite(str(OUT_DIR / img_name), frame)
                
                # Sprawdzamy, czy to jest ten konkretny moment kopnięcia ('K')
                is_kick = bool(abs(count - kick_frame_idx) <= EVERY_N / 2)

                # Zapisujemy do CSV TYLKO i WYŁĄCZNIE moment kopnięcia
                if is_kick:
                    frames_data.append({
                        "source_video": video_path.name,
                        "kick_image_name": img_name,
                        "kick_time_in_window_s": round(kick_in_window_s, 3)
                    })
                
                saved += 1
            count += 1
        
        cap.release()
        print(f"[OK] Z pliku {video_path.name} -> Zapisano {saved} zdjec")

    # Zapisujemy całą wiedzę do nowego pliku CSV
    df_meta = pd.DataFrame(frames_data)
    df_meta.to_csv(META_OUT, index=False)
    
    print(f"\nSUKCES! Twoje zdjecia (JPG) sa gotowe w folderze: {OUT_DIR.absolute()}")
    print(f"Mapa wszystkich zdjec (z zaznaczonym KICKIEM) zapisana w: {META_OUT.absolute()}")

if __name__ == "__main__":
    main()