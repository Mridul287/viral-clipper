# 🎬 Viral Clipper AI

An AI-powered video analysis tool that automatically extracts the most **viral-worthy clips** from long-form videos — podcasts, lectures, interviews, and more.

![Viral Clipper Banner](src/assets/hero.png)

---

## ✨ Features

- **AI Viral Scoring** — Scores video segments based on emotion, speech energy, scene changes, and semantic hooks
- **Auto Transcription** — Whisper-powered speech-to-text with word-level timestamps
- **Emotion Detection** — Frame-by-frame facial emotion analysis (excited, happy, neutral, etc.)
- **Scene Detection** — Automatically detects visual cuts and transitions
- **Clip Export** — Trims and exports the top-ranked clips with burned-in captions
- **AI Analyzer Dashboard** — Real-time emotion intensity bars synced to video playback
- **Futuristic UI** — Dark glassmorphism design with smooth animations

---

## 🧱 Tech Stack

| Layer | Tech |
|-------|------|
| **Frontend** | React 19, Vite, Framer Motion, Lucide React |
| **Backend** | FastAPI, Python 3.13 |
| **AI / ML** | OpenAI Whisper, OpenCV, MediaPipe, Ollama |
| **Video Processing** | FFmpeg (via `imageio-ffmpeg`), MoviePy |
| **Styling** | Vanilla CSS (glassmorphism + dark theme) |

---

## 📁 Project Structure

```
viral_clipper/
├── backend/
│   ├── main.py              # FastAPI app & REST endpoints
│   ├── pipeline.py          # Orchestrates the full AI pipeline
│   ├── ingest.py            # Video ingestion & download (yt-dlp)
│   ├── transcribe.py        # Whisper transcription
│   ├── emotions.py          # Emotion detection per frame
│   ├── scenes.py            # Scene change detection (OpenCV)
│   ├── scoring.py           # Viral clip scoring engine
│   ├── export.py            # Clip trimming, caption burning, thumbnails
│   ├── orchestrator.py      # Multi-step job orchestration
│   ├── job_store.py         # In-memory job state tracking
│   └── thumbnail_generator.py
├── src/
│   ├── App.jsx              # Main React app with all views
│   ├── VideoAnalysisDashboard.jsx  # AI analyzer with live emotion bars
│   ├── index.css            # Global styles & design system
│   └── main.jsx
├── public/
├── index.html
├── vite.config.js           # Vite + /api proxy config
└── package.json
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **FFmpeg** (bundled automatically via `imageio-ffmpeg` — no manual install needed)
- **Ollama** (optional — for LLM-based scoring)

### 1. Clone the repository

```bash
git clone https://github.com/Mridul287/viral_clipper.git
cd viral_clipper
```

### 2. Set up the Python backend

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install fastapi uvicorn openai-whisper opencv-python mediapipe \
            moviepy imageio-ffmpeg yt-dlp requests
```

### 3. Run the backend server

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at **http://localhost:8000**

> API docs: http://localhost:8000/docs

### 4. Set up the frontend

```bash
# From the project root
npm install
npm run dev
```

The frontend will be available at **http://localhost:5173**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload a video file or paste a URL |
| `GET` | `/status/{job_id}` | Poll job processing status & progress |
| `GET` | `/results/{job_id}` | Get final scored clips & metadata |
| `GET` | `/health` | Check API, Ollama, and Whisper status |

### Example: Upload a video file

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@my_podcast.mp4"
```

### Example: Upload via URL

```bash
curl -X POST http://localhost:8000/upload \
  -F "url=https://youtube.com/watch?v=xyz"
```

---

## 🧠 How the Pipeline Works

```
Upload Video
     │
     ▼
Ingest & Download (yt-dlp)
     │
     ▼
Transcribe Audio (Whisper)
     │
     ▼
Detect Emotions (OpenCV / MediaPipe)
     │
     ▼
Detect Scene Changes (OpenCV)
     │
     ▼
Score Segments (viral score formula)
     │
     ▼
Export Top Clips (FFmpeg trim + caption burn)
     │
     ▼
Results available via /results/{job_id}
```

---

## 🖥️ UI Screens

| Screen | Description |
|--------|-------------|
| **Dashboard** | Overview of recent jobs and quick stats |
| **Upload** | Drag-and-drop file upload or paste a video URL |
| **Analysis Engine** | Real-time processing progress with step indicators |
| **Generated Clips** | Browse and download the top-ranked viral clips |
| **AI Analyzer** | Frame-by-frame emotion intensity bars synced to playback |
| **Settings** | Configure AI models and system preferences |

---

## ⚙️ Configuration

The Vite dev server proxies all `/api` requests to the FastAPI backend at `localhost:8000`. This is configured in `vite.config.js`:

```js
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ''),
    },
  },
}
```

---

## 📝 Notes

- The `backend/temp/` and `backend/output/` directories are created automatically at runtime and are excluded from git.
- Large model files (Whisper weights) are downloaded automatically on first use and cached by the library.
- Ollama is optional — the scoring pipeline degrades gracefully without it.

---

## 📄 License

MIT License — feel free to fork and build on top of this!
