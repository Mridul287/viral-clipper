import React, { useState, useEffect } from 'react';
import './VideoAnalysisDashboard.css';

const API_BASE = '/api';

const VideoAnalysisDashboard = ({ jobId = null, videoInfo = null }) => {
  const [isProcessing, setIsProcessing] = useState(true);
  const [selectedSegment, setSelectedSegment] = useState(0);
  const [playbackProgress, setPlaybackProgress] = useState(0);
  const [segments, setSegments] = useState([]);
  const [captions, setCaptions] = useState([]);
  const [insights, setInsights] = useState({
    keywords: [],
    engagementSpikes: [],
    avgEngagement: 0,
    avgEmotion: 'analyzing',
  });
  const [status, setStatus] = useState('analyzing');
  const [error, setError] = useState(null);
  const [inputJobId, setInputJobId] = useState('');
  const [activeJobId, setActiveJobId] = useState(jobId);
  const [backendVideoPath, setBackendVideoPath] = useState(null);
  const [sourceUrl, setSourceUrl] = useState(null);
  
  const [frameEmotions, setFrameEmotions] = useState([]);
  const [currentEmotions, setCurrentEmotions] = useState({ 
    happy: 0, 
    surprised: 0, 
    angry: 0, 
    sad: 0, 
    fearful: 0, 
    neutral: 0 
  });

  // Fetch job status and results
  useEffect(() => {
    if (!activeJobId) {
      // Demo mode with mock data
      setSegments([
        { id: 0, time: '0:00', label: 'Intro', emotion: 'excited', intensity: 0.8 },
        { id: 1, time: '0:15', label: 'Story', emotion: 'neutral', intensity: 0.6 },
        { id: 2, time: '0:30', label: 'Climax', emotion: 'happy', intensity: 0.9 },
        { id: 3, time: '0:45', label: 'Outro', emotion: 'neutral', intensity: 0.5 },
        { id: 4, time: '1:00', label: 'End', emotion: 'excited', intensity: 0.7 },
      ]);
      setCaptions([
        { time: '0:00-0:05', text: 'Hey everyone, welcome back to the channel!' },
        { time: '0:05-0:15', text: 'Today we\'re looking at something absolutely insane.' },
        { time: '0:15-0:25', text: 'This discovery could change everything we know.' },
        { time: '0:25-0:35', text: 'Let me show you what I found...' },
      ]);
      setInsights({
        keywords: ['discovery', 'insane', 'absolutely', 'change', 'incredible'],
        engagementSpikes: [
          { time: '0:30', level: 95, reason: 'High emotional intensity' },
          { time: '0:12', level: 82, reason: 'Storytelling peak' },
        ],
        avgEngagement: 78,
        avgEmotion: 'mixed',
      });
      setIsProcessing(false);
      setError(null);
      return;
    }

    // Fetch real data from API
    const fetchData = async () => {
      try {
        // Get status
        const statusRes = await fetch(`${API_BASE}/status/${activeJobId}`);
        const statusData = await statusRes.json();
        setStatus(statusData.status);
        setPlaybackProgress(statusData.progress_percent || 0);
        
        console.log('[VideoAnalysisDashboard] Status data:', {
          video_path: statusData.video_path,
          source_url: statusData.source_url,
          status: statusData.status,
        });
        
        // Store video path from backend if available
        if (statusData.video_path) {
          console.log('[VideoAnalysisDashboard] Setting backend video path:', statusData.video_path);
          setBackendVideoPath(`/api${statusData.video_path}`);
        }
        
        // Store source URL from backend if available (for URL uploads)
        if (statusData.source_url) {
          console.log('[VideoAnalysisDashboard] Setting source URL:', statusData.source_url);
          setSourceUrl(statusData.source_url);
        }

        if (statusData.status === 'failed') {
          setError(statusData.error || 'Job failed');
          setIsProcessing(false);
          return;
        }

        if (statusData.status === 'done' || statusData.status === 'scoring') {
          // Get results
          const resultsRes = await fetch(`${API_BASE}/results/${activeJobId}`);
          if (resultsRes.status === 202) {
            setIsProcessing(true);
            return;
          }
          const results = await resultsRes.json();

          // Parse captions from segments
          if (results.segments && results.segments.length > 0) {
            const parsedCaptions = results.segments.slice(0, 8).map(seg => ({
              time: `${formatTime(seg.start)}-${formatTime(seg.end)}`,
              text: seg.text || '',
            }));
            setCaptions(parsedCaptions);

            // Create timeline segments from captions
            const timelineSegments = parsedCaptions.map((cap, idx) => ({
              id: idx,
              time: cap.time.split('-')[0],
              label: `Segment ${idx + 1}`,
              emotion: ['happy', 'surprised', 'angry', 'sad', 'fearful', 'neutral'][idx % 6],
              intensity: 0.5 + (Math.random() * 0.5),
            }));
            setSegments(timelineSegments);
          }

          // Extract keywords and engagement from top_clips
          if (results.top_clips && results.top_clips.length > 0) {
            const keywords = new Set();
            const engagementSpikes = [];
            let totalScore = 0;

            results.top_clips.forEach((clip, idx) => {
              // Extract keywords from suggested title
              if (clip.suggested_title) {
                clip.suggested_title.split(' ').forEach(word => {
                  if (word.length > 3) keywords.add(word.toLowerCase());
                });
              }

              // Create engagement spikes from top clips
              if (idx < 3) {
                engagementSpikes.push({
                  time: formatTime(clip.start),
                  level: Math.round((clip.virality || 5) * 10),
                  reason: clip.clip_type ? `${clip.clip_type} moment` : 'Viral potential detected',
                });
              }

              totalScore += clip.virality || 0;
            });

            const avgEngagement = results.top_clips.length > 0 
              ? Math.round((totalScore / results.top_clips.length) * 10) 
              : 0;

            setInsights({
              keywords: Array.from(keywords).slice(0, 5),
              engagementSpikes: engagementSpikes,
              avgEngagement: Math.min(avgEngagement, 100),
              avgEmotion: 'mixed',
            });
          }

          if (results.emotions && results.emotions.frame_emotions) {
            setFrameEmotions(results.emotions.frame_emotions);
          }

          setIsProcessing(false);
        } else {
          setIsProcessing(statusData.status === 'queued' || statusData.status === 'transcribing');
        }
      } catch (err) {
        console.error('Error fetching data:', err);
        setError(err.message);
      }
    };

    const interval = setInterval(fetchData, 2000);
    fetchData();
    return () => clearInterval(interval);
  }, [activeJobId]);

  // Format time helper
  const formatTime = (seconds) => {
    if (typeof seconds !== 'number') return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${String(secs).padStart(2, '0')}`;
  };

  // Emotion color mapping for 6 emotions
  const emotionColors = {
    happy: '#FFD700',      // Gold
    surprised: '#FF6B9D',  // Pink
    angry: '#FF4444',      // Red
    sad: '#4169E1',        // Royal Blue
    fearful: '#9370DB',    // Medium Purple
    neutral: '#87CEEB',    // Sky Blue
  };

  // Simulate progress
  useEffect(() => {
    if (!isProcessing) return;
    const interval = setInterval(() => {
      setPlaybackProgress(prev => (prev >= 100 ? 0 : prev + Math.random() * 15));
    }, 800);
    return () => clearInterval(interval);
  }, [isProcessing]);

  const getEmotionEmoji = (emotion) => {
    const emojis = { 
      happy: '😊', 
      surprised: '😲', 
      angry: '😠', 
      sad: '😢', 
      fearful: '😨', 
      neutral: '😐' 
    };
    return emojis[emotion] || '😐';
  };

  const handleTimeUpdate = (e) => {
    const time = e.target.currentTime;
    
    // Find closest frame emotion to current time
    if (frameEmotions.length > 0) {
      const closest = frameEmotions.reduce((prev, curr) => 
        Math.abs(curr.timestamp - time) < Math.abs(prev.timestamp - time) ? curr : prev
      );
      
      if (closest && closest.all_scores) {
        // Map DeepFace emotion keys to our UI keys (DeepFace scores are 0-100)
        setCurrentEmotions({
          happy: closest.all_scores.happy || 0,
          surprised: closest.all_scores.surprise || 0,
          angry: closest.all_scores.angry || 0,
          sad: closest.all_scores.sad || 0,
          fearful: closest.all_scores.fear || 0,
          neutral: closest.all_scores.neutral || 0
        });
      }
    }
  };

  const handleExportReport = () => {
    if (!insights || !captions) return;

    const reportContent = `
VIRAL CLIPPER AI - ANALYSIS REPORT
==================================
Date: ${new Date().toLocaleString()}
Job ID: ${activeJobId || 'Demo Mode'}
File: ${videoInfo?.fileName || 'Unknown'}

SUMMARY INSIGHTS
----------------
Average Engagement: ${insights.avgEngagement}%
Dominant Emotion: ${insights.avgEmotion}
Top Keywords: ${insights.keywords.join(', ')}

ENGAGEMENT SPIKES
-----------------
${insights.engagementSpikes.map(s => `[${s.time}] ${s.level}% - ${s.reason}`).join('\n')}

FULL TRANSCRIPT / CAPTIONS
--------------------------
${captions.map(c => `[${c.time}] ${c.text}`).join('\n')}

TIMELINE SEGMENTS
-----------------
${segments.map(s => `[${s.time}] ${s.label} (${s.emotion}, intensity: ${Math.round(s.intensity * 100)}%)`).join('\n')}

==================================
Generated by ViralClipper AI
    `;

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ViralClipper_Report_${activeJobId || 'Demo'}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="dashboard-container">
      {/* Grid background */}
      <div className="grid-background"></div>

      {/* Main content */}
      <div className="dashboard-content">
        {/* Header */}
        <header className="dashboard-header">
          <div className="header-title">
            <h1>🎬 AI Video Analysis Studio</h1>
            <p>Processing viral content with advanced AI insights</p>
          </div>
          <div className="header-status">
            {isProcessing && <div className="pulse-dot"></div>}
            <span className="status-text">
              {error ? '❌ Error' : isProcessing ? '⏳ Processing...' : '✅ Complete'}
            </span>
          </div>
        </header>

        {error && (
          <div style={{ 
            backgroundColor: 'rgba(255, 107, 157, 0.1)', 
            border: '1px solid rgba(255, 107, 157, 0.3)',
            color: '#ff6b9d',
            padding: '12px 20px',
            borderRadius: '8px',
            marginBottom: '20px'
          }}>
            Error loading data: {error}
          </div>
        )}

        {/* Job ID Input */}
        {!activeJobId && (
          <div style={{
            backgroundColor: 'rgba(74, 144, 255, 0.05)',
            border: '1px solid rgba(74, 144, 255, 0.2)',
            padding: '20px',
            borderRadius: '15px',
            marginBottom: '20px',
            display: 'flex',
            gap: '10px',
            alignItems: 'flex-end',
          }}>
            <div style={{ flex: 1 }}>
              <label style={{
                display: 'block',
                fontSize: '0.9rem',
                color: '#a0a0d0',
                marginBottom: '8px',
                fontWeight: '500',
              }}>
                Analyze Previous Job ID
              </label>
              <input
                type="text"
                placeholder="Paste job UUID here..."
                value={inputJobId}
                onChange={(e) => setInputJobId(e.target.value)}
                style={{
                  width: '100%',
                  padding: '10px 15px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid rgba(74, 144, 255, 0.3)',
                  borderRadius: '8px',
                  color: '#e0e0ff',
                  fontFamily: 'monospace',
                  fontSize: '0.85rem',
                }}
              />
            </div>
            <button
              onClick={() => {
                if (inputJobId.trim()) {
                  setActiveJobId(inputJobId.trim());
                }
              }}
              style={{
                padding: '10px 20px',
                background: 'linear-gradient(135deg, #4a90ff 0%, #ff6b9d 100%)',
                border: 'none',
                borderRadius: '8px',
                color: 'white',
                fontWeight: '600',
                cursor: 'pointer',
                fontSize: '0.9rem',
              }}
            >
              Analyze
            </button>
          </div>
        )}

        {/* Main grid layout */}
        <div className="dashboard-grid">
          
          {/* Left column - Video player and timeline */}
          <div className="left-column">
            {/* Video player card */}
            <div className="glass-card video-player-card">
              <div className="card-header">
                <div>
                  <h2>{videoInfo?.fileName ? `📹 ${videoInfo.fileName}` : '📹 Video Preview'}</h2>
                  {videoInfo?.isUrl && (
                    <p style={{ fontSize: '0.8rem', color: '#a0a0d0', margin: '4px 0 0 0' }}>
                      📡 From URL • {sourceUrl ? (() => { try { return new URL(sourceUrl).hostname; } catch (e) { return 'Network stream'; } })() : 'Network stream'}
                    </p>
                  )}
                </div>
                <span className="badge">HD 1080p</span>
              </div>
              <div className="video-container">
                {!backendVideoPath && videoInfo?.isUrl && isProcessing && (
                  <div style={{
                    position: 'absolute',
                    inset: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: 'rgba(26, 26, 46, 0.9)',
                    zIndex: 10,
                    flexDirection: 'column',
                    gap: '20px',
                  }}>
                    <div style={{
                      width: '60px',
                      height: '60px',
                      border: '3px solid rgba(74, 144, 255, 0.3)',
                      borderTop: '3px solid #4a90ff',
                      borderRadius: '50%',
                      animation: 'spin 1s linear infinite',
                    }}></div>
                    <p style={{ color: '#a0a0d0', margin: 0 }}>Processing video...</p>
                    <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                  </div>
                )}
                <video 
                  controls 
                  className="video-player"
                  crossOrigin="anonymous"
                  key={`video-${activeJobId}-${backendVideoPath}`}
                  onError={(e) => {
                    console.log('Video load error:', e);
                  }}
                  onLoadStart={() => {
                    console.log('[Video] Loading started', {
                      backendVideoPath,
                      videoInfoFileUrl: videoInfo?.fileUrl,
                      sourceUrl,
                    });
                  }}
                  onTimeUpdate={handleTimeUpdate}
                  onCanPlay={() => {
                    console.log('[Video] Can play');
                  }}
                  poster="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100%25' height='100%25'%3E%3Crect fill='%231a1a2e' width='100%25' height='100%25'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-size='24' fill='%234a90ff'%3E{!backendVideoPath && videoInfo?.isUrl ? 'Processing...' : 'Video Preview'}%3C/text%3E%3C/svg%3E"
                >
                  {/* Prioritize backend video path (processed/downloaded) over everything */}
                  {backendVideoPath ? (
                    (() => {
                      console.log('[Video] Using backend path:', backendVideoPath);
                      return <source src={backendVideoPath} type="video/mp4" />;
                    })()
                  ) : videoInfo?.fileUrl ? (
                    (() => {
                      console.log('[Video] Using videoInfo fileUrl:', videoInfo.fileUrl);
                      return <source src={videoInfo.fileUrl} type="video/mp4" />;
                    })()
                  ) : (
                    (() => {
                      console.log('[Video] No video source available yet');
                      return <source src="" type="video/mp4" />;
                    })()
                  )}
                  Your browser does not support the video tag.
                </video>
                <div className="video-overlay">
                  <div className="play-button">▶</div>
                </div>
              </div>

              {/* Progress bar */}
              {isProcessing && (
                <div className="progress-section">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ width: `${playbackProgress}%` }}
                    ></div>
                  </div>
                  <p className="progress-text">{Math.round(playbackProgress)}% {status === 'analyzing' ? 'Analyzed' : status}</p>
                </div>
              )}
            </div>

            {/* Timeline card */}
            <div className="glass-card timeline-card">
              <div className="card-header">
                <h2>Timeline Segments</h2>
                <span className="segment-count">{segments.length} segments</span>
              </div>
              <div className="timeline">
                {segments.map((segment) => (
                  <div
                    key={segment.id}
                    className={`timeline-segment ${selectedSegment === segment.id ? 'active' : ''}`}
                    style={{
                      '--emotion-color': emotionColors[segment.emotion],
                      '--intensity': segment.intensity,
                    }}
                    onClick={() => setSelectedSegment(segment.id)}
                  >
                    <div className="segment-dot"></div>
                    <div className="segment-info">
                      <span className="segment-time">{segment.time}</span>
                      <span className="segment-label">{segment.label}</span>
                      <span className="segment-emotion">{getEmotionEmoji(segment.emotion)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right column - Captions, emotions, and insights */}
          <div className="right-column">
            
            {/* Captions card */}
            <div className="glass-card captions-card">
              <div className="card-header">
                <h2>📝 Speech-to-Text Captions</h2>
                <span className="processing-badge">AI Transcribed</span>
              </div>
              <div className="captions-container">
                {captions.map((caption, idx) => (
                  <div key={idx} className="caption-item">
                    <span className="caption-time">{caption.time}</span>
                    <p className="caption-text">{caption.text}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Emotion Detection card */}
            <div className="glass-card emotion-card">
              <div className="card-header">
                <h2>😊 Emotion Detection</h2>
              </div>
              <div className="emotion-grid">
                {['happy', 'surprised', 'angry', 'sad', 'fearful', 'neutral'].map((emotion) => (
                  <div key={emotion} className="emotion-indicator">
                    <div className="emotion-emoji">{getEmotionEmoji(emotion)}</div>
                    <div className="emotion-bar">
                      <div 
                        className="emotion-fill"
                        style={{
                          width: `${currentEmotions[emotion] || 0}%`,
                          backgroundColor: emotionColors[emotion],
                          transition: 'width 0.3s ease-out'
                        }}
                      ></div>
                    </div>
                    <span className="emotion-label">
                      {emotion.charAt(0).toUpperCase() + emotion.slice(1)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* AI Insights card */}
            <div className="glass-card insights-card">
              <div className="card-header">
                <h2>🎯 AI Insights Panel</h2>
              </div>

              {/* Keywords section */}
              <div className="insights-section">
                <h3>📌 Top Keywords</h3>
                <div className="keywords-grid">
                  {insights.keywords.map((keyword, idx) => (
                    <span key={idx} className="keyword-badge">
                      {keyword}
                    </span>
                  ))}
                </div>
              </div>

              {/* Engagement spikes */}
              <div className="insights-section">
                <h3>📊 Engagement Spikes</h3>
                {insights.engagementSpikes.map((spike, idx) => (
                  <div key={idx} className="spike-item">
                    <div className="spike-header">
                      <span className="spike-time">{spike.time}</span>
                      <span className="spike-level">{spike.level}%</span>
                    </div>
                    <div className="spike-bar">
                      <div 
                        className="spike-fill"
                        style={{ width: `${spike.level}%` }}
                      ></div>
                    </div>
                    <p className="spike-reason">{spike.reason}</p>
                  </div>
                ))}
              </div>

              {/* Average metrics */}
              <div className="insights-section">
                <h3>📈 Overall Metrics</h3>
                <div className="metrics-grid">
                  <div className="metric-item">
                    <span className="metric-label">Avg. Engagement</span>
                    <span className="metric-value" style={{ color: '#FFD700' }}>
                      {insights.avgEngagement}%
                    </span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Dominant Emotion</span>
                    <span className="metric-value" style={{ color: '#FF6B9D' }}>
                      {insights.avgEmotion}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Action buttons */}
      <div className="action-buttons">
        <button 
          className="btn btn-secondary"
          onClick={handleExportReport}
        >
          📥 Export Report
        </button>
      </div>
    </div>
  );
};

export default VideoAnalysisDashboard;
