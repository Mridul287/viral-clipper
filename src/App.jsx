import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  UploadCloud, FileVideo, LayoutDashboard, FolderKanban, 
  Settings, Bell, Search, User, X, CheckCircle, Video,
  Play, Pause, SkipBack, SkipForward, Maximize, Mic,
  Activity, Zap, Target, BarChart2, MessageSquare, FastForward,
  Film, Download, Share2, Flame, Scissors, MoreHorizontal, Link,
  AlertCircle, Wifi, WifiOff, Cpu, Repeat2
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import VideoAnalysisDashboard from './VideoAnalysisDashboard';

// ---------------------------------------------------------------------------
// API helper — all requests go through Vite's /api proxy → localhost:8000
// ---------------------------------------------------------------------------
const API_BASE = '/api';

async function apiUploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

async function apiUploadUrl(url) {
  const form = new FormData();
  form.append('url', url);
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`URL upload failed: ${res.status}`);
  return res.json();
}

async function apiGetStatus(jobId) {
  const res = await fetch(`${API_BASE}/status/${jobId}`);
  if (!res.ok) throw new Error(`Status fetch failed: ${res.status}`);
  return res.json();
}

async function apiGetResults(jobId) {
  const res = await fetch(`${API_BASE}/results/${jobId}`);
  if (!res.ok) throw new Error(`Results fetch failed: ${res.status}`);
  return res.json();
}

async function apiHealth() {
  const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

// ---------------------------------------------------------------------------
// Utility: poll status until done/failed, calling onUpdate each tick
// ---------------------------------------------------------------------------
function usePollStatus(jobId, onUpdate, onComplete, onError) {
  const intervalRef = useRef(null);

  useEffect(() => {
    if (!jobId) return;

    const poll = async () => {
      try {
        const data = await apiGetStatus(jobId);
        onUpdate(data);
        if (data.status === 'done' || data.status === 'failed') {
          clearInterval(intervalRef.current);
          if (data.status === 'done') onComplete(data);
          else onError(data.error || 'Pipeline failed');
        }
      } catch (err) {
        clearInterval(intervalRef.current);
        onError(err.message);
      }
    };

    poll(); // immediate first call
    intervalRef.current = setInterval(poll, 2000);
    return () => clearInterval(intervalRef.current);
  }, [jobId]); // eslint-disable-line react-hooks/exhaustive-deps
}

// ---------------------------------------------------------------------------
// Backend Health Indicator (shown in header)
// ---------------------------------------------------------------------------
const BackendStatus = () => {
  const [health, setHealth] = useState(null); // null=loading, object=data, 'error'=down

  useEffect(() => {
    const check = async () => {
      try {
        const data = await apiHealth();
        setHealth(data);
      } catch {
        setHealth('error');
      }
    };
    check();
    const id = setInterval(check, 10000);
    return () => clearInterval(id);
  }, []);

  if (health === null) return null;

  const isUp = health !== 'error' && health?.status === 'ok';
  return (
    <div
      title={isUp ? `Backend: OK | Ollama: ${health.ollama_connected ? 'connected' : 'offline'} | Whisper: ${health.whisper_loaded ? 'loaded' : 'not loaded'}` : 'Backend offline'}
      style={{
        display: 'flex', alignItems: 'center', gap: '6px',
        padding: '4px 10px', borderRadius: '20px',
        background: isUp ? 'rgba(16,185,129,0.12)' : 'rgba(239,68,68,0.12)',
        border: `1px solid ${isUp ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}`,
        fontSize: '0.75rem', color: isUp ? '#10b981' : '#ef4444',
        cursor: 'default', userSelect: 'none',
      }}
    >
      {isUp ? <Wifi size={13} /> : <WifiOff size={13} />}
      {isUp ? 'API Live' : 'API Offline'}
    </div>
  );
};

// ---------------------------------------------------------------------------
// 1. UPLOAD VIEW — real file + URL uploads via /api/upload
// ---------------------------------------------------------------------------
const UploadView = ({ onUploadComplete }) => {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('idle'); // idle | uploading | complete | error
  const [errorMsg, setErrorMsg] = useState('');
  const [urlValue, setUrlValue] = useState('');
  const [isUrlLoading, setIsUrlLoading] = useState(false);
  const fileInputRef = useRef(null);
  const progressIntervalRef = useRef(null);

  const resetState = () => {
    setFile(null);
    setUploadStatus('idle');
    setUploadProgress(0);
    setErrorMsg('');
    setUrlValue('');
    if (fileInputRef.current) fileInputRef.current.value = '';
    clearInterval(progressIntervalRef.current);
  };

  // Animate progress bar to ~90% while real upload happens, then snap to 100
  const startFakeProgress = () => {
    setUploadProgress(0);
    progressIntervalRef.current = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 90) { clearInterval(progressIntervalRef.current); return 90; }
        return prev + Math.floor(Math.random() * 8) + 3;
      });
    }, 180);
  };

  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };

  const doFileUpload = async (selectedFile) => {
    setFile(selectedFile);
    setUploadStatus('uploading');
    setErrorMsg('');
    startFakeProgress();
    try {
      const data = await apiUploadFile(selectedFile);
      clearInterval(progressIntervalRef.current);
      setUploadProgress(100);
      setUploadStatus('complete');
      // Pass job_id and file info to parent
      onUploadComplete(data.job_id, {
        fileName: selectedFile.name,
        fileUrl: URL.createObjectURL(selectedFile),
      });
    } catch (err) {
      clearInterval(progressIntervalRef.current);
      setUploadStatus('error');
      setErrorMsg(err.message);
    }
  };

  const doUrlUpload = async () => {
    if (!urlValue.trim()) return;
    setIsUrlLoading(true);
    setErrorMsg('');
    try {
      const data = await apiUploadUrl(urlValue.trim());
      setIsUrlLoading(false);
      // Extract filename from URL or use default
      const urlObj = new URL(urlValue.trim());
      const pathname = urlObj.pathname;
      const fileName = pathname.substring(pathname.lastIndexOf('/') + 1) || 'video from URL';
      
      // For URL uploads, don't pass the URL as fileUrl (CORS issues with YouTube, etc.)
      // Only pass metadata, backend will process and serve the video
      onUploadComplete(data.job_id, {
        fileName: fileName,
        fileUrl: null, // Don't use original URL due to CORS
        isUrl: true,
      });
    } catch (err) {
      setIsUrlLoading(false);
      setErrorMsg(err.message);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) doFileUpload(e.dataTransfer.files[0]);
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) doFileUpload(e.target.files[0]);
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="page-content"
    >
      <div className="content-header" style={{ marginBottom: '2rem' }}>
        <h2>New AI Processing Task</h2>
        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Upload a podcast, lecture, or long-form video to extract viral clips.</p>
      </div>

      <AnimatePresence mode="wait">
        {uploadStatus === 'idle' && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            transition={{ duration: 0.3 }}
            className="upload-view-content"
          >
            <div 
              className={`upload-area ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <input 
                ref={fileInputRef}
                type="file" 
                accept="video/*,audio/*" 
                onChange={handleChange} 
                style={{ display: 'none' }} 
              />
              
              <div className="upload-icon-wrapper">
                <div className="upload-icon-glow"></div>
                <div className="upload-icon">
                  <UploadCloud size={40} />
                </div>
              </div>
              
              <h3 className="upload-title">Drag &amp; drop your video here</h3>
              <p className="upload-subtitle">or click to browse from your computer</p>
              
              <button className="btn-primary" onClick={(e) => { e.stopPropagation(); fileInputRef.current.click(); }}>
                Select File
              </button>
              
              <div className="supported-formats">
                Supported formats: MP4, MOV, AVI, WEBM, MP3, WAV &bull; Max file size: 500&nbsp;MB
              </div>
            </div>

            <div className="upload-divider">
              <span>OR</span>
            </div>

            <div className="url-upload-section glass-card">
              <div className="url-input-wrapper">
                <div className="url-icon"><Link size={20} /></div>
                <input 
                  type="text" 
                  placeholder="Paste YouTube, Vimeo, or Video URL here..." 
                  className="url-input-field"
                  value={urlValue}
                  onChange={(e) => setUrlValue(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && doUrlUpload()}
                />
                <button 
                  className="btn-primary url-submit-btn" 
                  onClick={(e) => { e.stopPropagation(); doUrlUpload(); }}
                  disabled={isUrlLoading || !urlValue.trim()}
                >
                  {isUrlLoading ? 'Fetching…' : 'Fetch Video'}
                </button>
              </div>
              {errorMsg && (
                <div style={{ display:'flex', alignItems:'center', gap:'6px', marginTop:'0.75rem', color:'#f87171', fontSize:'0.85rem' }}>
                  <AlertCircle size={15}/> {errorMsg}
                </div>
              )}
            </div>
          </motion.div>
        )}

        {(uploadStatus === 'uploading' || uploadStatus === 'complete' || uploadStatus === 'error') && file && (
          <motion.div 
            className="upload-progress-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <div className="video-thumbnail">
              <img src="https://images.unsplash.com/photo-1590845947385-bd0713364f9b?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80" alt="Video thumbnail preview" />
              <div className="thumbnail-overlay">
                <FileVideo size={24} color="white" />
              </div>
            </div>
            
            <div className="upload-details">
              <div className="upload-details-header">
                <div>
                  <div className="file-name">{file.name}</div>
                  <div className="file-size">{(file.size / (1024 * 1024)).toFixed(2)} MB</div>
                </div>
                {uploadStatus === 'uploading' ? (
                  <button className="icon-btn close-btn" onClick={resetState}>
                    <X size={20} />
                  </button>
                ) : uploadStatus === 'error' ? (
                  <button className="icon-btn close-btn" onClick={resetState} title="Retry">
                    <AlertCircle size={20} color="#f87171" />
                  </button>
                ) : (
                  <motion.div 
                    className="success-icon"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: 'spring', stiffness: 200, damping: 10 }}
                  >
                    <CheckCircle size={24} color="#10b981" />
                  </motion.div>
                )}
              </div>
              
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ 
                    width: `${uploadProgress}%`, 
                    background: uploadStatus === 'complete' ? '#10b981' : uploadStatus === 'error' ? '#ef4444' : '' 
                  }}
                ></div>
              </div>
              
              <div className="progress-status">
                <span style={{ color: uploadStatus === 'complete' ? '#10b981' : uploadStatus === 'error' ? '#f87171' : 'var(--neon-cyan)' }}>
                  {uploadStatus === 'complete' && 'Upload Complete — queued for processing'}
                  {uploadStatus === 'error' && (errorMsg || 'Upload failed')}
                  {uploadStatus === 'uploading' && `Uploading... ${uploadProgress}%`}
                </span>
                {uploadStatus === 'uploading' && <span>~ {Math.ceil((100 - uploadProgress) / 5)}s remaining</span>}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {uploadStatus === 'error' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}
          >
            <button className="btn-secondary" onClick={resetState}>Try Again</button>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

// ---------------------------------------------------------------------------
// 2. ANALYSIS DASHBOARD — polls /api/status, transitions when done
// ---------------------------------------------------------------------------
const AnalysisDashboardView = ({ jobId, videoInfo, onProcessingComplete, onShowAnalyzer }) => {
  const [jobStatus, setJobStatus] = useState({ status: 'queued', progress_percent: 0 });
  const [error, setError] = useState(null);

  const displayUrl = jobStatus?.video_path ? `/api${jobStatus.video_path}` : videoInfo?.fileUrl;

  // Map backend status labels to user-friendly step names
  const STEP_LABELS = {
    queued:       ['Queued',       'Waiting',    'Pending'],
    transcribing: ['Extracting',   'Transcribing', 'Pending'],
    scoring:      ['Extracted',    'Transcribed',  'Scoring'],
    done:         ['Extracted',    'Transcribed',  'Generated'],
    failed:       ['Error',        '—',            '—'],
  };

  const stepLabels = STEP_LABELS[jobStatus.status] || STEP_LABELS.queued;
  const activeStep = { queued: 0, transcribing: 1, scoring: 2, done: 3, failed: 3 }[jobStatus.status] ?? 0;

  usePollStatus(
    jobId,
    (data) => setJobStatus(data),
    () => onProcessingComplete(), // navigate to clips view
    (err) => setError(err),
  );

  // Show spinner / progress while processing
  const isProcessing = jobStatus.status !== 'done' && jobStatus.status !== 'failed';

  if (error) {
    return (
      <div className="processing-container">
        <div style={{ textAlign: 'center', color: '#f87171' }}>
          <AlertCircle size={48} style={{ marginBottom: '1rem' }} />
          <div style={{ fontSize: '1.1rem', fontWeight: 600 }}>Processing Failed</div>
          <div style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', fontSize: '0.9rem' }}>{error}</div>
        </div>
      </div>
    );
  }

  if (isProcessing) {
    return (
      <div className="processing-container">
        <div className="processing-rings">
          <div className="ring ring-1"></div>
          <div className="ring ring-2"></div>
          <div className="ring ring-3"></div>
        </div>
        
        <motion.div className="processing-status"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <div className="scramble-text">AI IS ANALYZING THE VIDEO...</div>
          <div className="progress-bar" style={{ width: '300px', margin: '0 auto 1.5rem', height: '4px', background: 'rgba(255,255,255,0.1)' }}>
             <motion.div 
               className="progress-fill"
               style={{ width: `${jobStatus.progress_percent}%` }}
               animate={{ width: `${jobStatus.progress_percent}%` }}
               transition={{ duration: 0.8, ease: 'easeOut' }}
             ></motion.div>
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginBottom: '1rem' }}>
            {jobStatus.progress_percent}% &bull; Job&nbsp;{jobId?.slice(0,8)}
          </div>
          <div className="processing-steps">
             <span className={`step ${activeStep >= 1 ? 'active' : ''}`}>{stepLabels[0]}</span>
             <span className={`step ${activeStep >= 2 ? 'active' : ''}`}>{stepLabels[1]}</span>
             <span className={`step ${activeStep >= 3 ? 'active' : ''}`}>{stepLabels[2]}</span>
          </div>
        </motion.div>
      </div>
    );
  }

  // Done — render the analysis grid (same as before, UI unchanged)
  return (
    <motion.div 
      className="dashboard-grid analysis-grid"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <div className="dashboard-col-main">
        <div className="glass-card video-card">
           <div className="video-container">
             {displayUrl ? (
               <video
                 controls
                 className="video-frame"
                 crossOrigin="anonymous"
                 style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                 key={displayUrl}
               >
                 <source src={displayUrl} type="video/mp4" />
                 Your browser does not support the video tag.
               </video>
             ) : (
               <img src="https://images.unsplash.com/photo-1516280440502-861f67825b59?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" alt="Video Player" className="video-frame" />
             )}
             <div className="video-overlay">
                {!displayUrl && (
                  <motion.div 
                    className="ai-bounding-box" 
                    style={{ left: '40%', top: '25%', width: '200px', height: '220px' }}
                    animate={{ left: ['40%', '42%', '39%', '40%'], top: ['25%', '24%', '26%', '25%'] }}
                    transition={{ duration: 6, repeat: Infinity, ease: 'linear' }}
                  >
                    <div className="corner-tl"></div>
                    <div className="corner-tr"></div>
                    <div className="corner-bl"></div>
                    <div className="corner-br"></div>
                    <div className="box-label">Excited (94%)</div>
                  </motion.div>
                )}
             </div>
             {!displayUrl && (
               <div className="video-controls-bar">
                  <Play fill="white" size={18} style={{cursor: 'pointer'}} />
                  <div className="video-time">02:14 / 45:20</div>
                  <div className="video-progress">
                    <div className="video-progress-fill" style={{ width: '4.8%' }}></div>
                  </div>
                  <Settings size={18} style={{cursor: 'pointer'}} />
                  <Maximize size={18} style={{cursor: 'pointer'}} />
               </div>
             )}
           </div>
           
           <div className="timeline-container">
             <div className="timeline-header">
               <div className="timeline-title"><Activity size={16}/> Engagement Timeline</div>
               <div className="timeline-legend">
                 <span className="legend-item"><div className="legend-dot spike-color"></div> Viral Spike</span>
                 <span className="legend-item"><div className="legend-dot hook-color"></div> Hook</span>
               </div>
             </div>
             
             <div className="timeline-bars">
                {Array.from({length: 60}).map((_, i) => {
                  let height = Math.random() * 30 + 15;
                  let type = 'neutral';
                  if (i > 10 && i < 15) { height = Math.random() * 40 + 50; type = 'spike'; }
                  if (i > 32 && i < 35) { height = Math.random() * 20 + 80; type = 'hook'; }
                  if (i > 45 && i < 52) { height = Math.random() * 30 + 60; type = 'spike'; }
                  
                  return (
                    <motion.div 
                      key={i} 
                      className={`timeline-bar type-${type}`} 
                      style={{ height: `${height}%` }}
                      initial={{ height: 0 }}
                      animate={{ height: `${height}%` }}
                      transition={{ duration: 0.5, delay: i * 0.01 }}
                    >
                      {type !== 'neutral' && <div className="bar-glow"></div>}
                      {i === 2 && <div className="playhead-line"></div>}
                    </motion.div>
                  );
                })}
             </div>
           </div>
        </div>

        <div className="glass-card emotion-card">
          <h3 className="card-title"><Zap size={18} className="neon-icon"/> Automated Emotion Detection</h3>
          <div className="emotion-gauges">
             <div className="gauge-item">
               <div className="radial-progress excited" style={{ '--progress': '78%' }}>
                  <div className="gauge-inner">78%</div>
               </div>
               <div className="gauge-label">Excited</div>
             </div>
             <div className="gauge-item">
               <div className="radial-progress happy" style={{ '--progress': '65%' }}>
                  <div className="gauge-inner">65%</div>
               </div>
               <div className="gauge-label">Happy</div>
             </div>
             <div className="gauge-item">
               <div className="radial-progress neutral" style={{ '--progress': '45%' }}>
                  <div className="gauge-inner">45%</div>
               </div>
               <div className="gauge-label">Neutral</div>
             </div>
          </div>
        </div>
      </div>

      <div className="dashboard-col-side">
        <div className="glass-card transcript-card">
           <div className="card-header">
             <h3 className="card-title no-border" style={{ marginBottom: 0, paddingBottom: 0 }}><Mic size={18} className="neon-icon"/> Live Transcript</h3>
             <span className="status-badge live">Sync</span>
           </div>
           
           <div className="transcript-scroll" style={{ marginTop: '1.5rem' }}>
              <p className="transcript-line time-stamp">02:10 - 02:14</p>
              <p className="transcript-line active">
                So when we think about the future of AI in marketing, it's not just about 
                <span className="highlight-word"> automation</span>. 
                It's about <span className="highlight-word viral">hyper-personalization</span> at an unimaginable scale.
              </p>
              
              <p className="transcript-line time-stamp">02:15 - 02:22</p>
              <p className="transcript-line">
                Exactly. And the matrix radically copies elements forever capturing
                people's attention near zero.
              </p>
           </div>
        </div>

        <div className="glass-card insights-card">
           <h3 className="card-title"><Target size={18} className="neon-icon"/> AI Driven Insights</h3>
           
           <div className="insight-section">
             <h4 className="insight-subtitle">Predicted Engagement</h4>
             <div className="engagement-score">
                <div className="score-value text-gradient">94<span className="score-out">/100</span></div>
                <div className="score-trend positive">↑ 12% vs channel average</div>
             </div>
             
             <div className="insight-section" style={{ marginTop: '1.5rem', marginBottom: '1.5rem' }}>
               <h4 className="insight-subtitle">Detected Keywords</h4>
               <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                 <span className="badge-success" style={{ background: 'rgba(147,51,234,0.1)', color: '#c084fc', border: '1px solid rgba(147,51,234,0.3)' }}>#AI</span>
                 <span className="badge-success" style={{ background: 'rgba(6,182,212,0.1)', color: 'var(--neon-cyan)', border: '1px solid rgba(6,182,212,0.3)' }}>#Automation</span>
                 <span className="badge-success" style={{ background: 'rgba(219,39,119,0.1)', color: '#f472b6', border: '1px solid rgba(219,39,119,0.3)' }}>#FutureTech</span>
                 <span className="badge-success" style={{ background: 'rgba(255,255,255,0.1)', color: 'white', border: '1px solid rgba(255,255,255,0.2)' }}>#Marketing</span>
               </div>
             </div>

             <div className="clips-suggested">
                <h4 className="insight-subtitle" style={{marginTop: '0.5rem'}}>Top Generated Clips</h4>
                <div className="clip-row" style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem', background: 'rgba(255,255,255,0.02)', padding: '0.5rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div className="clip-thumb glow-purple" style={{ position: 'relative', width: '80px', height: '45px', borderRadius: '4px', overflow: 'hidden' }}>
                    <img src="https://images.unsplash.com/photo-1543132220-4bf3de6e10ae?ixlib=rb-4.0.3&auto=format&fit=crop&w=100&q=80" alt="Clip 1" style={{ width: '100%', height: '100%', objectFit: 'cover' }}/>
                    <div className="clip-len" style={{ position: 'absolute', bottom: '2px', right: '4px', fontSize: '0.65rem', background: 'rgba(0,0,0,0.7)', padding: '1px 4px', borderRadius: '2px' }}>0:25</div>
                  </div>
                  <div className="clip-info" style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <div className="clip-h" style={{ fontSize: '0.9rem', fontWeight: 500 }}>"The AI Loophole"</div>
                    <div className="clip-score" style={{ fontSize: '0.8rem', color: 'var(--neon-purple)' }}>Viral Prob: 98%</div>
                  </div>
                </div>
             </div>
           </div>
        </div>
      </div>
      
      {/* AI Analyzer Button */}
      {onShowAnalyzer && (
        <motion.div 
          style={{ textAlign: 'center', marginTop: '20px' }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <motion.button 
            onClick={onShowAnalyzer}
            className="btn-primary"
            style={{ padding: '12px 32px', fontSize: '0.95rem' }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
          >
            <Cpu size={16} style={{ marginRight: '8px', display: 'inline' }} />
            View AI Analyzer Dashboard
          </motion.button>
        </motion.div>
      )}
    </motion.div>
  );
};

// ---------------------------------------------------------------------------
// 3. CLIPS VIEW — fetches real results from /api/results/{jobId}
// ---------------------------------------------------------------------------

// Utility: convert score 0–10 to a 0–100 percentage for display
const scoreToPercent = (score) => Math.round(Math.min(Math.max(score * 10, 0), 100));

// Utility: format seconds → m:ss
const fmtTime = (secs) => {
  const s = Math.round(secs);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};

// Clip Player Modal Component
const ClipPlayerModal = ({ clip, isOpen, onClose }) => {
  if (!isOpen || !clip) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
        }}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
          style={{
            background: 'linear-gradient(135deg, #1e1b4b 0%, #2d1b69 100%)',
            borderRadius: '16px',
            padding: '24px',
            maxWidth: '800px',
            maxHeight: '90vh',
            overflow: 'auto',
            border: '1px solid rgba(139, 92, 246, 0.3)',
          }}
        >
          {/* Close button */}
          <button
            onClick={onClose}
            style={{
              position: 'absolute',
              top: '16px',
              right: '16px',
              background: 'rgba(255, 255, 255, 0.1)',
              border: 'none',
              borderRadius: '50%',
              width: '40px',
              height: '40px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              color: 'white',
            }}
          >
            <X size={24} />
          </button>

          {/* Thumbnail */}
          <img
            src={clip.thumbnail_url}
            alt={clip.suggested_title}
            style={{
              width: '100%',
              borderRadius: '12px',
              marginBottom: '20px',
              maxHeight: '400px',
              objectFit: 'cover',
            }}
          />

          {/* Title */}
          <h2 style={{ fontSize: '1.8rem', fontWeight: 700, marginBottom: '12px', color: 'white' }}>
            {clip.suggested_title}
          </h2>

          {/* Scores */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px', marginBottom: '20px' }}>
            <div style={{ background: 'rgba(139, 92, 246, 0.2)', padding: '12px', borderRadius: '8px' }}>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Viral Score</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#a78bfa' }}>
                {scoreToPercent(clip.final_score)}%
              </div>
            </div>
            <div style={{ background: 'rgba(139, 92, 246, 0.2)', padding: '12px', borderRadius: '8px' }}>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Duration</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#a78bfa' }}>
                {fmtTime(clip.end - clip.start)}
              </div>
            </div>
          </div>

          {/* Clip Details */}
          <div style={{ background: 'rgba(0, 0, 0, 0.3)', padding: '16px', borderRadius: '8px', marginBottom: '20px' }}>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '8px' }}>Clip Details</div>
            <div style={{ fontSize: '0.95rem', color: '#e0e7ff', lineHeight: '1.6' }}>
              <p><strong>Type:</strong> {clip.clip_type}</p>
              <p><strong>Starts at:</strong> {fmtTime(clip.start)}</p>
              <p><strong>Ends at:</strong> {fmtTime(clip.end)}</p>
              <p><strong>Segment text:</strong> {clip.text}</p>
            </div>
          </div>

          {/* Score Breakdown */}
          <div style={{ background: 'rgba(0, 0, 0, 0.3)', padding: '16px', borderRadius: '8px' }}>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '12px' }}>Score Breakdown</div>
            {Object.entries(clip.scores).map(([key, value]) => (
              <div key={key} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ textTransform: 'capitalize' }}>{key}</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '100px', height: '6px', background: 'rgba(139, 92, 246, 0.2)', borderRadius: '3px' }}>
                    <div
                      style={{
                        width: `${(value / 10) * 100}%`,
                        height: '100%',
                        background: 'linear-gradient(90deg, #8b5cf6, #a78bfa)',
                        borderRadius: '3px',
                      }}
                    />
                  </div>
                  <span style={{ minWidth: '25px' }}>{value}/10</span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

// Placeholder thumbnails (rotated by rank since we have no real frames)
const THUMB_POOL = [
  'https://images.unsplash.com/photo-1534528741775-53994a69daeb?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80',
  'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80',
  'https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80',
  'https://images.unsplash.com/photo-1524504388940-b1c1722653e1?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80',
  'https://images.unsplash.com/photo-1559523161-0fc0d8b38a7a?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80',
  'https://images.unsplash.com/photo-1581368135153-a506cf13b1e1?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80',
];

const ClipsView = ({ jobId }) => {
  const [clips, setClips] = useState(null);   // null = loading, [] = empty, [...] = results
  const [filter, setFilter] = useState('viral');
  const [loadError, setLoadError] = useState(null);
  const [selectedClip, setSelectedClip] = useState(null);

  useEffect(() => {
    if (!jobId) {
      // No job yet: show placeholder demo clips
      setClips([]);
      return;
    }
    apiGetResults(jobId)
      .then(data => {
        if (data.top_clips) setClips(data.top_clips);
        else setClips([]);
      })
      .catch(err => setLoadError(err.message));
  }, [jobId]);

  // Fallback demo clips shown when no job is active
  const demoClips = [
    { rank: 1, suggested_title: 'The AI Growth Loophole', start: 0, end: 45, final_score: 9.8, clip_type: 'hook' },
    { rank: 2, suggested_title: 'Why Optimization is Dead', start: 120, end: 152, final_score: 9.4, clip_type: 'key_fact' },
    { rank: 3, suggested_title: 'Hyper-personalization at Scale', start: 300, end: 359, final_score: 8.9, clip_type: 'motivational' },
    { rank: 4, suggested_title: 'The Organic Reach Shift', start: 500, end: 525, final_score: 8.4, clip_type: 'intense' },
    { rank: 5, suggested_title: 'Secret to Viewer Retention', start: 700, end: 718, final_score: 9.2, clip_type: 'funny' },
    { rank: 6, suggested_title: 'Automating The Future', start: 900, end: 950, final_score: 8.0, clip_type: 'key_fact' },
  ];

  const displayClips = (clips && clips.length > 0) ? clips : demoClips;

  const sorted = [...displayClips].sort((a, b) => {
    if (filter === 'viral') return b.final_score - a.final_score;
    if (filter === 'shortest') return (a.end - a.start) - (b.end - b.start);
    return a.rank - b.rank;
  });

  return (
    <motion.div 
      className="page-content clips-page"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      style={{ maxWidth: '1400px' }}
    >
      <div className="clips-header">
        <div>
          <h2 style={{ fontSize: '1.8rem', fontFamily: 'Outfit', fontWeight: 700, marginBottom: '0.25rem' }}>Generated Shorts</h2>
          <p style={{ color: 'var(--text-secondary)' }}>
            {loadError
              ? <span style={{ color: '#f87171' }}>⚠ {loadError}</span>
              : clips === null
              ? 'Loading results…'
              : `AI successfully extracted ${displayClips.length} viral clip${displayClips.length !== 1 ? 's' : ''}.${!jobId ? ' (Demo data)' : ''}`
            }
          </p>
        </div>
        
        <div className="filter-group">
          <button className={`filter-btn ${filter === 'viral' ? 'active' : ''}`} onClick={() => setFilter('viral')}>
            🔥 Most Viral
          </button>
          <button className={`filter-btn ${filter === 'shortest' ? 'active' : ''}`} onClick={() => setFilter('shortest')}>
            <Scissors size={14} style={{ display: 'inline', marginRight: '4px' }} /> Shortest
          </button>
          <button className={`filter-btn ${filter === 'newest' ? 'active' : ''}`} onClick={() => setFilter('newest')}>
            Newest
          </button>
        </div>
      </div>

      <div className="reels-grid">
        <AnimatePresence>
          {sorted.map((clip, index) => {
            const duration = clip.end - clip.start;
            const pct = scoreToPercent(clip.final_score);
            // Use real thumbnail if available, otherwise placeholder
            const thumb = clip.thumbnail_url || THUMB_POOL[index % THUMB_POOL.length];
            return (
              <motion.div 
                key={clip.rank} 
                className="reel-card" 
                layout
                initial={{ opacity: 0, scale: 0.9 }} 
                animate={{ opacity: 1, scale: 1 }} 
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                onClick={() => setSelectedClip(clip)}
                style={{ cursor: 'pointer' }}
              >
                <img src={thumb} alt={clip.suggested_title} className="reel-img" />
                <div className="reel-overlay">
                  <div className="viral-badge"><Flame size={14} /> {pct}%</div>
                  <div className="play-overlay"><Play size={24} fill="currentColor" color="white" style={{ marginLeft: '4px' }} /></div>
                  <div className="reel-bottom">
                    <h4 className="reel-title">"{clip.suggested_title}"</h4>
                    <div className="reel-meta-row">
                      <span className="reel-duration">{fmtTime(duration)}</span>
                      <div className="reel-actions">
                        <button className="reel-action-btn" title="Download"><Download size={16} /></button>
                        <button className="reel-action-btn" title="Share"><Share2 size={16} /></button>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {/* Clip Player Modal */}
      <ClipPlayerModal 
        clip={selectedClip} 
        isOpen={!!selectedClip} 
        onClose={() => setSelectedClip(null)} 
      />
    </motion.div>
  );
};

// ---------------------------------------------------------------------------
// 4. USER DASHBOARD VIEW (unchanged UI, wires "New Project" nav)
// ---------------------------------------------------------------------------
const UserDashboardView = ({ onUploadClick }) => {
  return (
    <motion.div 
      className="page-content user-dashboard-wrapper"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="dashboard-header-flex">
        <div>
          <h2 style={{ fontSize: '2rem', fontFamily: 'Outfit', fontWeight: 700 }}>Welcome back, Creator!</h2>
          <p style={{ color: 'var(--text-secondary)' }}>Here is what's happening with your content this week.</p>
        </div>
        <button className="btn-primary" onClick={onUploadClick} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <UploadCloud size={20} /> New Project
        </button>
      </div>

      {/* Stats Summary Panel */}
      <div className="stats-grid">
        <div className="glass-card stat-card">
          <div className="stat-icon glow-cyan"><Video size={24}/></div>
          <div className="stat-info">
             <div className="stat-value">124</div>
             <div className="stat-label">Videos Processed</div>
          </div>
        </div>
        <div className="glass-card stat-card">
          <div className="stat-icon glow-purple"><Scissors size={24}/></div>
          <div className="stat-info">
             <div className="stat-value">842</div>
             <div className="stat-label">Clips Generated</div>
          </div>
        </div>
        <div className="glass-card stat-card">
          <div className="stat-icon glow-pink"><Target size={24}/></div>
          <div className="stat-info">
             <div className="stat-value text-gradient">92/100</div>
             <div className="stat-label">Avg. Engagement Rating</div>
          </div>
        </div>
      </div>

      <div className="dashboard-main-grid">
        {/* Left Column: Activity Chart */}
        <div className="flex-col">
          <div className="glass-card chart-card">
             <div className="card-header" style={{ marginBottom: '1rem' }}>
                <h3 className="card-title no-border" style={{ margin: 0, padding: 0 }}><BarChart2 size={18} className="neon-icon"/> Activity Graph</h3>
                <span className="status-badge live" style={{ animation: 'none', background: 'rgba(255,255,255,0.05)', color: 'var(--text-secondary)', border: '1px solid rgba(255,255,255,0.1)' }}>This Week</span>
             </div>
             
             <div className="chart-container">
               <div className="chart-bars">
                 {[40, 60, 30, 85, 50, 95, 70].map((h, i) => (
                   <motion.div 
                     key={i} 
                     className="chart-bar-wrapper"
                     initial={{ opacity: 0, y: 20 }}
                     animate={{ opacity: 1, y: 0 }}
                     transition={{ delay: i * 0.1, duration: 0.4 }}
                   >
                     <div className="chart-bar" style={{height: `${h}%`}}></div>
                     <span className="chart-day">{'SMTWTFS'[i]}</span>
                   </motion.div>
                 ))}
               </div>
             </div>
          </div>
        </div>

        {/* Right Column: Recent Projects */}
        <div className="glass-card recent-projects-card flex-col" style={{ gap: 0 }}>
           <div className="card-header" style={{ marginBottom: '1.5rem' }}>
             <h3 className="card-title no-border" style={{ margin: 0, padding: 0 }}><FolderKanban size={18} className="neon-icon"/> Recent Projects</h3>
             <button className="icon-btn"><MoreHorizontal size={20}/></button>
           </div>
           
           <div className="projects-list">
              {[
                { name: 'Podcast_Ep42.mp4', time: '2 hours ago', clips: 12, status: 'Completed' },
                { name: 'Marketing_Webinar.mov', time: 'Yesterday', clips: 8, status: 'Completed' },
                { name: 'Tech_Interview.mp4', time: '3 days ago', clips: 25, status: 'Completed' },
                { name: 'Product_Launch.mp4', time: '5 days ago', clips: 18, status: 'Completed' }
              ].map((p, i) => (
                <motion.div 
                  key={i} 
                  className="project-list-item"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 + (i * 0.1) }}
                >
                   <div className="project-icon"><Video size={16}/></div>
                   <div className="project-info">
                      <div className="project-name">{p.name}</div>
                      <div className="project-meta">{p.clips} clips &bull; {p.time}</div>
                   </div>
                   <div className="project-status badge-success">
                      {p.status}
                   </div>
                </motion.div>
              ))}
           </div>
        </div>
      </div>
    </motion.div>
  );
};

// ---------------------------------------------------------------------------
// 5. SETTINGS VIEW (unchanged)
// ---------------------------------------------------------------------------
const SettingsView = () => {
  const [notifications, setNotifications] = useState(true);
  const [autoSave, setAutoSave] = useState(true);
  const [highContrast, setHighContrast] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [showSavedToast, setShowSavedToast] = useState(false);

  const handleSave = () => {
    setIsSaving(true);
    setTimeout(() => {
      setIsSaving(false);
      setShowSavedToast(true);
      setTimeout(() => setShowSavedToast(false), 3000);
    }, 1200);
  };

  const containerVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.6, staggerChildren: 0.1 } }
  };
  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  return (
    <motion.div 
      className="page-content settings-page-premium"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Settings Header & Profile Card */}
      <motion.div className="settings-header-card glass-card" variants={itemVariants}>
        <div className="settings-banner"></div>
        <div className="settings-profile-info">
          <div className="avatar-wrapper-premium">
            <img src="https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?ixlib=rb-4.0.3&auto=format&fit=crop&w=100&q=80" alt="Profile" />
          </div>
          <div className="profile-text-premium">
            <h2>Mridul Tripathi</h2>
            <p>Creative Director &amp; AI Specialist</p>
          </div>
          <div style={{ flex: 1 }}></div>
          <button className="btn-secondary" style={{ padding: '0.5rem 1rem' }}>Edit Bio</button>
        </div>
      </motion.div>

      <div className="settings-main-layout">
        <div className="settings-sidebar-nav">
          <div className="sidebar-nav-link active">General</div>
          <div className="sidebar-nav-link">Security</div>
          <div className="sidebar-nav-link">Billing</div>
          <div className="sidebar-nav-link">API Keys</div>
        </div>

        <div className="settings-fields-container">
          {/* Profile Section */}
          <motion.div className="glass-card settings-premium-card" variants={itemVariants}>
            <div className="premium-card-header">
              <User size={18} className="neon-icon" />
              <h3>Account Information</h3>
            </div>
            <div className="premium-form-grid">
              <div className="form-group-premium">
                <label>Display Name</label>
                <input type="text" placeholder="Your Name" defaultValue="Mridul Tripathi" />
              </div>
              <div className="form-group-premium">
                <label>Email Address</label>
                <input type="email" placeholder="email@example.com" defaultValue="mridul@viralclipper.ai" />
              </div>
            </div>
          </motion.div>

          {/* Preferences Section */}
          <motion.div className="glass-card settings-premium-card" variants={itemVariants}>
            <div className="premium-card-header">
              <Zap size={18} className="neon-icon" />
              <h3>Workflow Preferences</h3>
            </div>
            <div className="premium-form-grid">
              <div className="form-group-premium">
                <label>Rendering Quality</label>
                <select className="select-premium">
                  <option>4K Ultra HD (Recommended)</option>
                  <option>1080p High Definition</option>
                  <option>720p Standard</option>
                </select>
              </div>
              <div className="form-group-premium">
                <label>Subtitle AI Model</label>
                <select className="select-premium">
                  <option>Vision-X Dynamic</option>
                  <option>Neural-Overlay Classic</option>
                  <option>Ghost-Text Minimal</option>
                </select>
              </div>
            </div>
          </motion.div>

          {/* System Controls */}
          <motion.div className="glass-card settings-premium-card" variants={itemVariants}>
            <div className="premium-card-header">
              <Bell size={18} className="neon-icon" />
              <h3>System Controls</h3>
            </div>
            <div className="toggles-container-premium">
              <div className="premium-toggle-row">
                <div className="toggle-text-block">
                  <div className="toggle-title-premium">Smart Notifications</div>
                  <div className="toggle-subtitle-premium">AI will only alert you for major viral breakthroughs.</div>
                </div>
                <label className="ios-switch">
                  <input type="checkbox" checked={notifications} onChange={() => setNotifications(!notifications)} />
                  <span className="ios-slider"></span>
                </label>
              </div>

              <div className="premium-toggle-row">
                <div className="toggle-text-block">
                  <div className="toggle-title-premium">Instant Cloud Sync</div>
                  <div className="toggle-subtitle-premium">Automatically push generated clips to TikTok &amp; Instagram drafts.</div>
                </div>
                <label className="ios-switch">
                  <input type="checkbox" checked={autoSave} onChange={() => setAutoSave(!autoSave)} />
                  <span className="ios-slider"></span>
                </label>
              </div>

              <div className="premium-toggle-row">
                <div className="toggle-text-block">
                  <div className="toggle-title-premium">Energy Saving Mode</div>
                  <div className="toggle-subtitle-premium">Reduces UI animations to save GPU resources during processing.</div>
                </div>
                <label className="ios-switch">
                  <input type="checkbox" checked={highContrast} onChange={() => setHighContrast(!highContrast)} />
                  <span className="ios-slider"></span>
                </label>
              </div>
            </div>
          </motion.div>

          <div className="settings-premium-footer">
            <AnimatePresence>
              {showSavedToast && (
                <motion.div 
                  className="saved-toast"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                >
                  <CheckCircle size={16} /> Changes saved successfully
                </motion.div>
              )}
            </AnimatePresence>
            <div style={{ flex: 1 }}></div>
            <button className="btn-secondary">Discard</button>
            <motion.button 
              className="btn-primary premium-save-btn" 
              onClick={handleSave}
              disabled={isSaving}
              whileTap={{ scale: 0.98 }}
            >
              {isSaving ? 'Saving...' : 'Save Configuration'}
            </motion.button>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// ---------------------------------------------------------------------------
// MAIN APP — wires job state through the entire flow
// ---------------------------------------------------------------------------
function App() {
  const [currentView, setCurrentView] = useState('user-dashboard');
  // jobId is passed from UploadView → AnalysisDashboardView → ClipsView
  const [activeJobId, setActiveJobId] = useState(null);
  const [videoInfo, setVideoInfo] = useState(null); // Store video file info

  // Called when /upload returns — navigate to analysis and carry the job_id
  const handleUploadComplete = (jobId, fileInfo = null) => {
    setActiveJobId(jobId);
    setVideoInfo(fileInfo);
    setCurrentView('analysis');
  };

  // Called when polling confirms job is done — navigate to clips
  const handleProcessingComplete = () => {
    setCurrentView('clips');
  };

  return (
    <div className="app-container">
      <div className="bg-grid"></div>
      <div className="bg-glow"></div>

      {/* Sidebar */}
      <aside className="sidebar">
        <a href="#" onClick={(e) => {e.preventDefault(); setCurrentView('user-dashboard');}} className="sidebar-logo">
          <div className="logo-icon"></div>
          <div className="logo-text" style={{ fontSize: '1.25rem' }}>ViralClipper <span>AI</span></div>
        </a>

        <nav className="sidebar-nav">
          <a href="#" onClick={(e) => { e.preventDefault(); setCurrentView('user-dashboard'); }} className={`nav-item ${currentView === 'user-dashboard' ? 'active' : ''}`}>
            <LayoutDashboard size={20} />
            <span>Dashboard</span>
          </a>
          <a href="#" onClick={(e) => { e.preventDefault(); setCurrentView('upload'); }} className={`nav-item ${currentView === 'upload' || currentView === 'analysis' ? 'active' : ''}`}>
            <UploadCloud size={20} />
            <span>Upload</span>
          </a>
          <a href="#" onClick={(e) => { e.preventDefault(); setCurrentView('clips'); }} className={`nav-item ${currentView === 'clips' ? 'active' : ''}`}>
            <Film size={20} />
            <span>Generated Clips</span>
          </a>
          <a href="#" onClick={(e) => { e.preventDefault(); setCurrentView('analyzer'); }} className={`nav-item ${currentView === 'analyzer' ? 'active' : ''}`}>
            <Cpu size={20} />
            <span>AI Analyzer</span>
          </a>
          <a href="#" onClick={(e) => { e.preventDefault(); setCurrentView('settings'); }} className={`nav-item ${currentView === 'settings' ? 'active' : ''}`}>
            <Settings size={20} />
            <span>Settings</span>
          </a>
          <div style={{ flex: 1 }}></div>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="top-header">
          <div className="page-title">
            <h1 className="brand-font" style={{ fontSize: '1.4rem', fontWeight: 600 }}>
              {currentView === 'user-dashboard' && 'Welcome'}
              {currentView === 'upload' && 'Upload Video'}
              {currentView === 'analysis' && 'Analysis Engine'}
              {currentView === 'clips' && 'AI Content Results'}
              {currentView === 'analyzer' && 'AI Video Analysis'}
              {currentView === 'settings' && 'Settings'}
            </h1>
          </div>
          
          <div className="header-actions">
            <div className="search-bar">
              <Search size={16} />
              <input type="text" placeholder="Search projects..." />
            </div>
            <BackendStatus />
            <button className="icon-btn"><Bell size={20} /></button>
            <div className="user-profile">
              <img src="https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?ixlib=rb-4.0.3&auto=format&fit=crop&w=100&q=80" alt="User profile icon" />
            </div>
          </div>
        </header>

        {currentView === 'user-dashboard' && <UserDashboardView key="dashboard" onUploadClick={() => setCurrentView('upload')} />}
        {currentView === 'upload' && <UploadView key="upload" onUploadComplete={handleUploadComplete} />}
        {currentView === 'analysis' && (
          <AnalysisDashboardView
            key="analysis"
            jobId={activeJobId}
            videoInfo={videoInfo}
            onProcessingComplete={handleProcessingComplete}
            onShowAnalyzer={() => setCurrentView('analyzer')}
          />
        )}
        {currentView === 'clips' && <ClipsView key="clips" jobId={activeJobId} />}
        {currentView === 'analyzer' && <VideoAnalysisDashboard key="analyzer" jobId={activeJobId} videoInfo={videoInfo} />}
        {currentView === 'settings' && <SettingsView key="settings" />}
      </main>
    </div>
  );
}

export default App;
