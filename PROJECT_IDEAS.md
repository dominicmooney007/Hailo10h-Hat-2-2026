# Hailo AI Hat+2 (Hailo-10H) Student Project Ideas

## What Makes the Hailo-10H Special

The Hailo-10H is unique because it combines **edge computer vision** (detection, pose, segmentation, depth, face recognition, OCR, CLIP) with **on-device generative AI** (LLM, VLM, Whisper speech-to-text, TTS). No cloud needed. This is the killer differentiator — students can build fully autonomous, private, intelligent systems on a Raspberry Pi.

---

## Project Ideas (Ranked by "Wow Factor" for Demos)

---

### 1. "JARVIS" — Voice-Controlled Vision Assistant
**Combines:** Voice Assistant + VLM + Detection + Depth + OCR + CLIP + LLM Agent Tools + LED feedback

**What it does:** A hands-free AI assistant that can *see* and *speak*. Ask it "What's in front of me?" and it describes the scene via VLM. Ask "How far is that chair?" and it uses depth estimation. Say "Read that sign" and it runs OCR. Say "Find something red" and CLIP searches the frame. Say "Turn on the light" and the agent framework controls the LED.

**Why students love it:** It's Iron Man's JARVIS on a Pi. Combines nearly every capability into one cohesive demo. Great for accessibility applications too.

**Key learning:** Multi-model orchestration, voice pipelines, agent tool calling, real-time inference chaining.

**Difficulty:** Hard | **Features used:** 8+

---

### 2. Smart Doorbell / Visitor Recognition System
**Combines:** Face Recognition + LanceDB + Detection + Depth + Telegram Alerts + Voice Assistant + LED indicators

**What it does:** Recognizes family/friends by face (trained via face recognition pipeline + LanceDB). Unknown visitors trigger a Telegram photo alert with distance estimate. The system greets known people by name using TTS. NeoPixel LEDs show status (green = known, red = unknown, blue = processing). Voice interaction lets visitors leave a message transcribed by Whisper.

**Why students love it:** Immediately practical — everyone understands a smart doorbell. Shows real-world AI product thinking. Parents/friends can be enrolled in the face DB live during the demo.

**Key learning:** Face embedding databases, vector similarity search, notification systems, multi-stage inference pipelines.

**Difficulty:** Medium | **Features used:** 6+

---

### 3. AI Sports Coach / Exercise Form Checker
**Combines:** Pose Estimation + LLM + Voice Feedback (TTS) + Tracking + Depth

**What it does:** Watches a person exercise (squats, pushups, jumping jacks) via pose estimation. Tracks joint angles frame-by-frame. Counts reps automatically. Detects bad form (e.g., knees caving in on squats, back arching on pushups). Uses depth to gauge distance/positioning. Sends pose data to the on-device LLM which generates natural-language coaching feedback, spoken aloud via TTS: *"Good rep! But try to keep your knees over your toes."*

**Why students love it:** Physical, interactive, immediately testable. Great for live demos — have audience members try exercises.

**Key learning:** Keypoint geometry, angle calculation, temporal analysis, LLM prompt engineering for domain-specific feedback.

**Difficulty:** Medium | **Features used:** 5

---

### 4. Real-Time Lecture/Whiteboard Digitizer
**Combines:** OCR + VLM + Whisper (Speech-to-Text) + LLM + Tiling + Super Resolution

**What it does:** Point a camera at a whiteboard/chalkboard during a lecture. OCR extracts handwritten text in real-time. Tiling handles the full-resolution board. VLM interprets diagrams and equations ("This appears to be a circuit diagram with..."). Whisper transcribes the lecturer's speech simultaneously. The LLM merges visual + audio content into structured notes with timestamps. Output: auto-generated lecture notes combining what was said and what was written.

**Why students love it:** Solves a real problem they face every day. "Never miss a lecture note again." Combines vision + audio + language in a way that feels magical.

**Key learning:** Multi-modal fusion, OCR pipelines, speech processing, LLM summarization, high-resolution image handling with tiling.

**Difficulty:** Hard | **Features used:** 6

---

### 5. Wildlife / Pet Monitoring Camera
**Combines:** Detection + CLIP + Tracking + Tiling + Segmentation + LanceDB + Telegram + VLM

**What it does:** An outdoor (or pet) camera that detects animals/pets, uses CLIP for zero-shot species classification ("Is this a cat, dog, fox, squirrel, bird?"), tracks individuals across frames, segments them from the background, and logs sightings to a LanceDB database. Telegram alerts for rare/new sightings with VLM-generated descriptions: *"A red fox was spotted near the garden at 14:32, appearing to investigate the compost bin."*

**Why students love it:** Animals are universally engaging. Can be demoed with pet videos or wildlife footage. Shows zero-shot learning (no training needed for new species).

**Key learning:** Zero-shot classification with CLIP, tracking persistence, database logging, alert systems, instance segmentation for clean captures.

**Difficulty:** Medium | **Features used:** 7

---

### 6. Multi-Camera Security System with Cross-Camera Tracking
**Combines:** Multi-source + ReID + Face Recognition + Detection + Tracking + Depth + Telegram + LanceDB

**What it does:** 2-3 USB cameras covering different "zones." Detects and tracks people across all cameras simultaneously using ReID. Recognizes known faces. Tracks movement patterns (Person entered Zone A at 10:01, moved to Zone B at 10:03). Depth estimation gauges proximity to cameras. Unknown faces trigger Telegram alerts. Dashboard shows real-time multi-camera feed with tracking overlays.

**Why students love it:** Multi-camera setups look professional and impressive. Cross-camera tracking feels like sci-fi surveillance tech. Great for discussing AI ethics too.

**Key learning:** Multi-stream processing, cross-camera re-identification, distributed inference, GStreamer pipeline architecture.

**Difficulty:** Medium-Hard | **Features used:** 7

---

### 7. Intelligent Inventory / Shelf Scanner
**Combines:** Detection + Tiling + CLIP + OCR + LLM + Voice Interface + LanceDB

**What it does:** Scan a shelf/fridge/pantry. Tiling handles the wide-angle high-res image. Detection finds objects, CLIP classifies them zero-shot ("cereal box," "milk carton," "apple"). OCR reads labels and expiry dates. LLM generates a natural-language inventory report and answers questions: *"You have 3 apples, milk expiring tomorrow, and you're out of bread."* Voice interface for hands-free queries: "Do I have enough eggs for a cake?"

**Why students love it:** Practical daily-life application. Easy to demo with any shelf of items. Shows how multiple AI models compose into something genuinely useful.

**Key learning:** Multi-model composition, tiling for high-res, zero-shot vs trained classification, structured data extraction.

**Difficulty:** Medium | **Features used:** 6

---

### 8. Gesture-Controlled Hardware Lab
**Combines:** Pose Estimation + Depth + Servo Control + LED Control + LLM Agent + Voice

**What it does:** Control physical hardware with your body. Raise your right hand = servo moves right. Left hand up = LED turns blue. Clap (detected via pose) = reset. Depth determines intensity (closer hand = brighter LED, further = dimmer). Voice commands for complex sequences: "Sweep the servo back and forth 3 times." LLM agent interprets natural language into tool calls.

**Why students love it:** Physical computing meets AI. Instant visual feedback. Feels like controlling things with superpowers. Great for younger students / outreach events.

**Key learning:** Pose keypoint mapping to control signals, depth-based continuous control, agent tool framework, real-time inference latency considerations.

**Difficulty:** Medium | **Features used:** 6

---

### 9. Accessible Navigation Aid (AI for Good)
**Combines:** Detection + Depth + Segmentation + VLM + Voice (TTS + Whisper) + CLIP

**What it does:** A wearable/portable vision system for visually impaired users. Depth estimation warns of obstacles ("Object 2 meters ahead"). Detection identifies what it is ("Person approaching on your left"). Segmentation maps walkable areas. VLM provides rich scene descriptions on demand. CLIP answers specific queries ("Is there a door nearby?"). All output via TTS. All input via Whisper voice commands.

**Why students love it:** Meaningful "AI for Good" project. Demonstrates that edge AI can genuinely help people. Excellent for grant applications and competitions. Sparks important conversations about responsible AI.

**Key learning:** Multi-model fusion for safety-critical applications, latency requirements, accessible interface design, ethical AI considerations.

**Difficulty:** Hard | **Features used:** 7

---

### 10. Live Video Game / Interactive AR Experience
**Combines:** Detection + Pose + Segmentation + Depth + Tracking + LLM + TTS + LED

**What it does:** An augmented-reality-style game where players interact with virtual elements overlaid on the camera feed. Pose estimation tracks player movement (dodge, jump, punch). Detection spawns "enemies" when real objects appear. Segmentation creates player silhouettes. Depth determines game layers. LLM narrates the game: *"A wild goblin appears! Strike a pose to attack!"* LEDs flash for hits/misses. Tracking maintains game state across frames.

**Why students love it:** It's a video game powered by AI. Extremely engaging for demos and open days. Appeals to students who wouldn't normally be interested in "serious" AI.

**Key learning:** Real-time game loop design, combining multiple inference streams, creative applications of CV, latency optimization.

**Difficulty:** Hard | **Features used:** 8

---

## Quick-Win Demo Projects (Single Feature Showcases)

These are simpler projects great for first-time exploration or short workshops:

| Project | Primary Feature | Time to Build |
|---|---|---|
| **"What Am I Holding?"** — Point objects at camera, CLIP identifies them from text prompts | CLIP | 1-2 hours |
| **Attendance Tracker** — Face recognition logs who enters a room with timestamps | Face Recognition + LanceDB | 2-3 hours |
| **Crowd Counter** — Count people in a space using detection + tiling for wide angles | Detection + Tiling | 1-2 hours |
| **Sign Language Speller** — Use pose estimation hand keypoints to recognize basic gestures | Pose Estimation | 3-4 hours |
| **Speed Reader** — Point camera at a book, OCR extracts text, LLM summarizes it, TTS reads aloud | OCR + LLM + TTS | 2-3 hours |
| **Mood Ring Camera** — Detect faces, use VLM to assess expressions, LED changes color to match | VLM + Face Detection + LED | 2-3 hours |
| **Parking Spot Finder** — Detect cars in a parking lot image (tiling), count empty spaces | Detection + Tiling + Segmentation | 2-3 hours |
| **Voice-Controlled Object Finder** — "Where are my keys?" → detection + CLIP + TTS response | Voice + CLIP + Detection + TTS | 3-4 hours |

---

## Recommended "Hero Demo" for Maximum Impact

If you can only build **one** project to showcase the Hailo-10H to students, I'd recommend **Project 1 ("JARVIS")** or **Project 3 (AI Sports Coach)**:

- **JARVIS** shows the broadest range of capabilities and feels futuristic
- **AI Sports Coach** is the most *interactive* and physically engaging for live demos — have students do squats and get real-time AI feedback

Both demonstrate the Hailo-10H's unique selling point: **vision + language + speech, all on-device, no cloud, real-time**.

---

## Available Hailo-10H Capabilities Reference

| Category | Features | Models |
|---|---|---|
| Object Detection | 20+ YOLO variants | YOLOv5/6/7/8/9/10/11 |
| Pose Estimation | 17-point skeleton | YOLOv8 Pose |
| Instance Segmentation | Object masks | YOLOv5/8 Seg, FastSAM |
| Semantic Segmentation | Pixel-level classification | FCN8, SegFormer, STDC1, DeepLab |
| Depth Estimation | Monocular + Stereo | SCDepthV3, StereoNet |
| Face Detection | Anchor-based face detection | SCRFD 10g / 2.5g |
| Face Recognition | Embedding-based matching | ArcFace MobileFaceNet |
| CLIP | Zero-shot classification | CLIP ViT-B/32 |
| OCR | Text detection + recognition | PaddleOCR (DB + CRNN) |
| LLM | On-device text generation | Qwen2.5-1.5B-Instruct |
| VLM | Vision-language understanding | Qwen2-VL-2B-Instruct |
| Speech-to-Text | Audio transcription | Whisper-Base |
| Text-to-Speech | Voice synthesis | Piper TTS |
| Tracking | Multi-object tracking | ByteTrack + Kalman Filter |
| Tiling | High-res small object detection | Multi-scale grid processing |
| Multi-Camera | Parallel stream processing | HailoRoundRobin / StreamRouter |
| ReID | Cross-camera person tracking | ArcFace + SCRFD |
| Hardware Control | GPIO/PWM/SPI | Servo motors, NeoPixel LEDs |
| Notifications | Push alerts with images | Telegram Bot API |
| Vector Database | Embedding storage/search | LanceDB |
| Agent Framework | LLM function calling | Tool discovery + execution |
| Lane Detection | Road lane marking | UFLD V2 TU |
| Super Resolution | Image upscaling | Real ESRGAN x2 |

---

## Architecture Principle for All Projects

```
Camera → GStreamer Pipeline → Hailo-10H Inference → Post-Processing → Application Logic
                                    ↓                                        ↓
                              Multiple Models                    LLM/VLM/Voice (GenAI)
                              (Detection, Pose,                  Agent Tools (Hardware)
                               Depth, CLIP, etc.)               Database (LanceDB)
                                                                 Alerts (Telegram)
```

Every project follows this pattern. The magic is in how you **compose** the building blocks.
