import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  UploadCloud, 
  Smartphone, 
  CheckCircle, 
  AlertCircle, 
  Download,
  Link,
  X,
  FileVideo
} from 'lucide-react';

const API_BASE = '/api';

export default function ReframeView() {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [status, setStatus] = useState('idle'); // idle | uploading | reframing | complete | error
  const [jobId, setJobId] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [urlValue, setUrlValue] = useState('');
  const [isUrlLoading, setIsUrlLoading] = useState(false);
  const [resultData, setResultData] = useState(null);
  const fileInputRef = useRef(null);
  const progressIntervalRef = useRef(null);

  const resetState = () => {
    setFile(null);
    setJobId(null);
    setStatus('idle');
    setUploadProgress(0);
    setResultData(null);
    setErrorMsg('');
    setUrlValue('');
    if (fileInputRef.current) fileInputRef.current.value = '';
    clearInterval(progressIntervalRef.current);
    setIsUrlLoading(false);
  };

  const startFakeProgress = () => {
    setUploadProgress(0);
    progressIntervalRef.current = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 90) { clearInterval(progressIntervalRef.current); return 90; }
        return prev + Math.floor(Math.random() * 8) + 3;
      });
    }, 200);
  };

  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
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

  const doFileUpload = async (selectedFile) => {
    setFile(selectedFile);
    setStatus('uploading');
    setErrorMsg('');
    startFakeProgress();

    const form = new FormData();
    form.append('file', selectedFile);

    try {
      const res = await fetch(`${API_BASE}/reframe`, { method: 'POST', body: form });
      clearInterval(progressIntervalRef.current);
      setUploadProgress(100);

      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
      const data = await res.json();
      
      setJobId(data.job_id);
      setStatus('reframing');
    } catch (err) {
      clearInterval(progressIntervalRef.current);
      setStatus('error');
      setErrorMsg(err.message);
    }
  };

  const doUrlUpload = async () => {
    if (!urlValue.trim()) return;
    setIsUrlLoading(true);
    setErrorMsg('');
    setStatus('uploading');
    startFakeProgress();

    const form = new FormData();
    form.append('url', urlValue.trim());

    try {
      const res = await fetch(`${API_BASE}/reframe`, { method: 'POST', body: form });
      clearInterval(progressIntervalRef.current);
      setUploadProgress(100);

      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
      const data = await res.json();
      
      setJobId(data.job_id);
      setStatus('reframing');
      setIsUrlLoading(false);
    } catch (err) {
      clearInterval(progressIntervalRef.current);
      setIsUrlLoading(false);
      setStatus('error');
      setErrorMsg(err.message);
    }
  };

  // Poll job status
  useEffect(() => {
    if (status !== 'reframing' || !jobId) return;

    const intervalId = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/status/${jobId}`);
        const data = await res.json();

        if (data.status === 'done') {
          clearInterval(intervalId);
          const resultRes = await fetch(`${API_BASE}/results/${jobId}`);
          const payload = await resultRes.json();
          setResultData(payload);
          setStatus('complete');
        } else if (data.status === 'failed') {
          clearInterval(intervalId);
          setStatus('error');
          setErrorMsg(data.error || 'Reframing failed');
        }
      } catch (err) {
        clearInterval(intervalId);
        setStatus('error');
        setErrorMsg('Lost connection to server');
      }
    }, 2000);

    return () => clearInterval(intervalId);
  }, [status, jobId]);

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="page-content"
    >
      <div className="content-header" style={{ marginBottom: '2rem' }}>
        <h2><Smartphone size={24} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '8px', paddingBottom: '3px' }}/>Auto-Reframe to 9:16</h2>
        <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Upload any horizontal video. Our AI FaceTracker will lock onto the speaker and perfectly frame them for TikTok & Reels.</p>
      </div>

      <AnimatePresence mode="wait">
        {status === 'idle' && (
          <motion.div 
            key="idle"
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
                accept="video/*" 
                onChange={handleChange} 
                style={{ display: 'none' }} 
              />
              
              <div className="upload-icon-wrapper">
                <div className="upload-icon-glow"></div>
                <div className="upload-icon">
                  <Smartphone size={40} />
                </div>
              </div>
              
              <h3 className="upload-title">Drag &amp; drop your video here to reframe</h3>
              <p className="upload-subtitle">or click to browse from your computer</p>
              
              <button className="btn-primary" onClick={(e) => { e.stopPropagation(); fileInputRef.current.click(); }}>
                Select File
              </button>
              
              <div className="supported-formats">
                Supported formats: MP4, MOV, AVI, WEBM &bull; Max file size: 500&nbsp;MB
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
                  {isUrlLoading ? 'Fetching...' : 'Fetch Video'}
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

        {(status === 'uploading' || status === 'reframing' || (status === 'error' && !resultData)) && (
          <motion.div 
            key="processing"
            className="upload-progress-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            style={{ maxWidth: '800px', margin: '0 auto' }}
          >
            <div className="video-thumbnail">
              <img src="https://images.unsplash.com/photo-1620121692029-d088224ddc74?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80" alt="Video thumbnail preview" />
              <div className="thumbnail-overlay">
                <FileVideo size={24} color="white" />
              </div>
            </div>
            
            <div className="upload-details">
              <div className="upload-details-header">
                <div>
                  <div className="file-name">{file ? file.name : (urlValue || 'Video Link')}</div>
                  <div className="file-size">{file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : 'Remote Source'}</div>
                </div>
                {status === 'error' ? (
                  <button className="icon-btn close-btn" onClick={resetState} title="Retry">
                    <AlertCircle size={20} color="#f87171" />
                  </button>
                ) : (
                  <button className="icon-btn close-btn" onClick={resetState}>
                    <X size={20} />
                  </button>
                )}
              </div>
              
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ 
                    width: status === 'reframing' ? '100%' : `${uploadProgress}%`, 
                    background: status === 'error' ? '#ef4444' : 'var(--neon-purple)',
                  }}
                ></div>
              </div>
              
              <div className="progress-status">
                <span style={{ color: status === 'error' ? '#f87171' : 'var(--neon-purple)' }}>
                  {status === 'error' && (errorMsg || 'Upload failed')}
                  {status === 'uploading' && `Uploading... ${uploadProgress}%`}
                  {status === 'reframing' && `Reframing video framing via AI tracking...`}
                </span>
                {status === 'uploading' && <span>~ {Math.ceil((100 - uploadProgress) / 5)}s remaining</span>}
              </div>
            </div>
          </motion.div>
        )}

        {status === 'complete' && resultData && (
          <motion.div 
            key="complete"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass-card"
            style={{ padding: '0', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}
          >
            <div style={{ padding: '2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <CheckCircle size={32} color="#10b981" />
                <div>
                  <h3 style={{ margin: 0, fontSize: '1.4rem' }}>Reframe Complete!</h3>
                </div>
              </div>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <a 
                  href={resultData.url} 
                  download={`reframed_${file ? file.name : 'video'}.mp4`}
                  className="btn-primary" 
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                >
                  <Download size={18} /> Download High-Res
                </a>
                <button className="btn-secondary" onClick={resetState}>
                  Reframe Another
                </button>
              </div>
            </div>

            <div style={{ background: '#000', width: '100%', display: 'flex', justifyContent: 'center' }}>
              <video 
                src={resultData.url} 
                controls 
                autoPlay
                loop
                style={{ width: '100%', height: '85vh', maxHeight: '1000px', objectFit: 'contain' }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
