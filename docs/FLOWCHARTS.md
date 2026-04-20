# Pipeline Flowcharts

All diagrams are in Mermaid format. Paste them into any Mermaid renderer
(e.g. mermaid.live, VS Code Mermaid plugin, Typora) to render visually.

---

## 1. Full Goalkeeper Goal-Line Pipeline

```mermaid
flowchart TD
    A([Penalty kick video clip\n720p broadcast footage]) --> B

    B[Automatic Kick Detection\nball motion velocity analysis] --> C

    C{Ball detected\nin window?}
    C -- Yes --> D[Kick frame estimate]
    C -- No --> FAIL([Pipeline failure\nno kick frame])

    D --> E[Frame adjustment\nkick_frame_adjust = -1]

    E --> F[Extract single frame\nat adjusted kick moment]

    F --> G[YOLOv8n Detection\ngoalkeeper + ball classes]

    G --> H{Goalkeeper\ndetected?}
    H -- No --> UNC1([uncertain\nno goalkeeper found])
    H -- Yes --> I[Goal-line detection\nHough + color filtering]

    I --> J{Goal line\ndetected?}
    J -- No --> UNC2([uncertain\nno line found])
    J -- Yes --> K[Foot-proxy extraction\nbbox bottom point]

    K --> L[Distance computation\nfoot proxy to goal line]

    L --> M[Uncertainty Policy\nboundary + local_y_err checks]

    M --> N{Uncertainty\nconditions met?}
    N -- Yes --> UNC3([uncertain\ntoo close to boundary\nor high geometry error])
    N -- No --> O{Distance\npositive?}

    O -- Goalkeeper behind line --> VALID([valid\non_line])
    O -- Goalkeeper in front of line --> VIOL([violation\noff_line])
```

---

## 2. Automatic Kick Detection Submodule

```mermaid
flowchart TD
    A([Video clip]) --> B

    B[Define analysis window\n0.5s to 2.5s from clip start] --> C

    C[Run YOLO ball detection\non every frame in window] --> D

    D[Build ball trajectory\nfilter low-confidence detections] --> E

    E{Enough ball\ndetections?}
    E -- No --> FAIL([kick detection failed\nfallback or error])
    E -- Yes --> F

    F[Compute frame-by-frame\nball velocity px/frame] --> G

    G[Compute baseline velocity\nmedian of stable period] --> H

    H[Detect onset of large\nvelocity increase\nmotion_onset method] --> I

    I{Onset\nfound?}
    I -- No --> PEAK[Fallback: use frame\nof peak velocity]
    I -- Yes --> J

    PEAK --> J[Raw kick frame estimate]

    J --> K[Apply frame adjustment\nraw_frame + kick_frame_adjust]

    K --> L{Adjusted frame\nin valid range?}
    L -- No --> CLAMP[Clamp to frame 0]
    L -- Yes --> OUT

    CLAMP --> OUT([Final kick frame index])
```

---

## 3. Goal-Line Decision Logic

```mermaid
flowchart TD
    A([YOLO detections\nfor kick frame]) --> B

    B[Select goalkeeper box\nhighest confidence detection] --> C

    C[Extract foot proxy point\nbottom-center of bbox] --> D

    D[Detect goal line\nHough transform on lower image region] --> E

    E{Line\nfound?}
    E -- No --> NOLN([uncertain — no line])
    E -- Yes --> F

    F[Fit line geometry\nto detected white horizontal line] --> G

    G[Project foot proxy\nonto line geometry] --> H

    H[Compute min_dist_px\nsigned distance foot → line] --> I

    I[Compute local_y_err_px\nvertical geometry estimation error] --> J

    J --> K{min_dist_px > 0?\ni.e. foot is in front of line?}
    K -- No / on line --> VALID([raw: on_line])
    K -- Yes --> VIOL([raw: off_line])
```

---

## 4. Uncertainty Policy

```mermaid
flowchart TD
    A([Raw decision + distances\nmin_dist_px, local_y_err_px]) --> B

    B{near_boundary?\nmin_dist_px < 2.0 px}
    B -- Yes --> FLAG1[flag: near_boundary]
    B -- No --> C

    FLAG1 --> C

    C{high_local_y_err?\nlocal_y_err_px > 8.0 px}
    C -- Yes --> FLAG2[flag: high_local_y_err]
    C -- No --> D

    FLAG2 --> D

    D{Any flags\nset?}
    D -- Yes --> UNC([uncertain\nwith flag reasons])
    D -- No --> KEEP([keep raw decision\non_line or off_line])
```

---

## 5. Extended Pipeline — Combined Officiating (Experimental)

```mermaid
flowchart TD
    A([Penalty kick video clip]) --> B & G

    B[Main pipeline\nGoalkeeper goal-line analysis] --> C
    C([GK decision:\nvalid / violation / uncertain])

    G[Encroachment probe\npre-kick frame analysis] --> H
    H[Detect all players\nin penalty area] --> I
    I[Check player positions\nrelative to penalty arc] --> J
    J([Encroachment flag:\nok / potential_encroachment / uncertain])

    C --> OUT
    J --> OUT
    OUT([Combined officiating output\nGK decision + encroachment flag])
```

---

## Usage notes

- Diagrams 1–4 cover the final adopted thesis method
- Diagram 5 covers the experimental extension (not part of the main thesis result)
- For the report: Diagram 1 goes in the Methods section overview
- For the report: Diagrams 2–4 go in the respective subsections
- For the presentation: Diagram 1 is the main "architecture" slide
