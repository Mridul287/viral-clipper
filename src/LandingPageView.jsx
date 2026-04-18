import React from 'react';
import { motion } from 'framer-motion';
import { Play, Sparkles, Scissors, TrendingUp, Captions, ChevronRight, Star } from 'lucide-react';
import './LandingPageView.css';

const LandingPageView = ({ onGetStarted }) => {
  return (
    <div className="landing-container">
      {/* Abstract Backgrounds */}
      <div className="landing-bg-glow glow-1"></div>
      <div className="landing-bg-glow glow-2"></div>
      <div className="landing-bg-grid"></div>

      {/* Navbar */}
      <nav className="landing-nav">
        <div className="landing-logo">
          <div className="landing-logo-icon"></div>
          ViralAI
        </div>
        <div className="landing-nav-links">
          <a href="#features">Features</a>
          <a href="#how-it-works">How it Works</a>
          <a href="#testimonials">Testimonials</a>
          <a href="#pricing">Pricing</a>
        </div>
        <div className="landing-nav-actions">
          <button className="landing-btn-text" onClick={onGetStarted}>Login</button>
          <button className="landing-btn-primary" onClick={onGetStarted}>Sign Up</button>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="landing-hero">
        <motion.div 
          className="hero-badge"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <Sparkles size={16} /> Welcome to the future of content creation
        </motion.div>
        
        <motion.h1 
          className="hero-title"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          Turn Long Videos into <br />
          <span className="text-gradient">Viral Shorts</span> using AI.
        </motion.h1>
        
        <motion.p 
          className="hero-subtitle"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          Our advanced AI analyzes emotions, detects semantic hooks, and automatically clips your podcast or lecture into perfectly framed, highly engaging vertical shorts.
        </motion.p>
        
        <motion.div 
          className="hero-cta-group"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <button className="landing-btn-primary large" onClick={onGetStarted}>
            Get Started <ChevronRight size={18} />
          </button>
          <button className="landing-btn-glass large" onClick={onGetStarted}>
            Upload Video
          </button>
        </motion.div>
        

      </section>

      {/* Features Section */}
      <section id="features" className="landing-features">
        <h2 className="section-title">AI-Powered Features</h2>
        <p className="section-subtitle">Everything you need to dominate short-form platforms</p>
        
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon"><Sparkles size={24} /></div>
            <h3>Deep AI Analysis</h3>
            <p>We analyze facial emotions and speech energy to find the most engaging highlights of your videos.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon"><Scissors size={24} /></div>
            <h3>Auto Face Tracking</h3>
            <p>Perfectly crops and centers speakers for 9:16 vertical shorts automatically.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon"><Captions size={24} /></div>
            <h3>Dynamic Subtitles</h3>
            <p>Generates highly accurate, animated subtitles with highlighted viral keywords.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon"><TrendingUp size={24} /></div>
            <h3>Viral Scoring Engine</h3>
            <p>Ranks your generated clips by their probability of going viral based on millions of data points.</p>
          </div>
        </div>
      </section>

      {/* How it Works Section */}
      <section id="how-it-works" className="landing-how-it-works">
        <h2 className="section-title">How It Works</h2>
        <p className="section-subtitle">From raw footage to viral clip in minutes</p>
        
        <div className="steps-container">
          <div className="step-item">
            <div className="step-number">1</div>
            <h3>Upload or Paste URL</h3>
            <p>Upload your local file or drop a YouTube link. We support videos up to 2 hours long.</p>
          </div>
          <div className="step-item">
            <div className="step-number">2</div>
            <h3>AI Orchestration</h3>
            <p>Our serverless GPUs transcribe, analyze emotions, and detect scenes in real-time.</p>
          </div>
          <div className="step-item">
            <div className="step-number">3</div>
            <h3>Download & Publish</h3>
            <p>Instantly download fully-edited shorts packed with captions, hooks, and engagement.</p>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section id="testimonials" className="landing-testimonials">
        <h2 className="section-title">Loved by Creators</h2>
        <div className="testimonials-grid">
          <div className="testimonial-card">
            <div className="stars"><Star fill="#facc15" color="#facc15"/><Star fill="#facc15" color="#facc15"/><Star fill="#facc15" color="#facc15"/><Star fill="#facc15" color="#facc15"/><Star fill="#facc15" color="#facc15"/></div>
            <p className="quote">"ViralAI completely transformed my workflow. What used to take me 5 hours in Premiere Pro now takes 5 minutes!"</p>
            <div className="author">
               <img src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&auto=format&fit=crop&w=100&q=80" alt="Avatar"/>
               <div>
                  <h4>Alex Chen</h4>
                  <span>Tech Podcaster</span>
               </div>
            </div>
          </div>
          <div className="testimonial-card">
            <div className="stars"><Star fill="#facc15" color="#facc15"/><Star fill="#facc15" color="#facc15"/><Star fill="#facc15" color="#facc15"/><Star fill="#facc15" color="#facc15"/><Star fill="#facc15" color="#facc15"/></div>
            <p className="quote">"The auto face-tracking and emotion radar are insane. My TikTok engagement has doubled since I started using this."</p>
            <div className="author">
               <img src="https://images.unsplash.com/photo-1494790108377-be9c29b29330?ixlib=rb-4.0.3&auto=format&fit=crop&w=100&q=80" alt="Avatar"/>
               <div>
                  <h4>Sarah Mitchell</h4>
                  <span>Content Creator</span>
               </div>
            </div>
          </div>
          <div className="testimonial-card">
            <div className="stars"><Star fill="#facc15" color="#facc15"/><Star fill="#facc15" color="#facc15"/><Star fill="#facc15" color="#facc15"/><Star fill="#facc15" color="#facc15"/><Star fill="#facc15" color="#facc15"/></div>
            <p className="quote">"The easiest and most accurate semantic clip picker out there. It literally chooses the best moments."</p>
            <div className="author">
               <img src="https://images.unsplash.com/photo-1500648767791-00dcc994a43e?ixlib=rb-4.0.3&auto=format&fit=crop&w=100&q=80" alt="Avatar"/>
               <div>
                  <h4>David Kim</h4>
                  <span>Agency Owner</span>
               </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-top">
          <div className="footer-brand">
            <div className="landing-logo-icon"></div>
            <h2>ViralAI</h2>
            <p>Empowering creators with AI.</p>
          </div>
          <div className="footer-links">
            <div className="link-col">
              <h4>Product</h4>
              <a href="#">Features</a>
              <a href="#">Pricing</a>
              <a href="#">API</a>
            </div>
            <div className="link-col">
              <h4>Company</h4>
              <a href="#">About Us</a>
              <a href="#">Careers</a>
              <a href="#">Blog</a>
            </div>
            <div className="link-col">
              <h4>Legal</h4>
              <a href="#">Privacy Policy</a>
              <a href="#">Terms of Service</a>
            </div>
          </div>
        </div>
        <div className="footer-bottom">
          &copy; {new Date().getFullYear()} ViralAI Inc. All rights reserved.
        </div>
      </footer>
    </div>
  );
};

export default LandingPageView;
