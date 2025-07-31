# Motion-Kombat

🚀 Project Overview : 
    I built a gesture-based control system for Mortal Kombat XL, enabling real-time gameplay using full-body movements via a webcam — no controller needed!

🧠 How It Works

    Used YOLOv8 Pose for real-time 17-point human pose estimation
    
    Mapped body keypoints to Mortal Kombat actions using OpenCV + vgamepad
    
    Supported features:
    
    Dynamic left/right facing mode toggle
    
    Punches with elbow gestures (X/Y)
    
    Special combos based on hand positions (Combo 1 & 2)
    
    Jump, crouch, move left/right using nose and hip detection
    
    A & B buttons triggered by hand-over-head gestures

⚙️ Tech Stack
    
    Python
    
    YOLOv8 (Ultralytics)
    
    OpenCV
    
    vgamepad (Xbox Virtual Gamepad Emulation)

Real-time pose detection and gesture classification logic
