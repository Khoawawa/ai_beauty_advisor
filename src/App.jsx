import React, { useState, useRef, useEffect } from 'react';
import './App.css';

export default function App() {
  const [activePage, setActivePage] = useState('home');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);
  const [productFilter, setProductFilter] = useState('all');
  const [compareList, setCompareList] = useState([]);
  const METRIC_PERCENTAGES = {
    // Undertone
    "Warm":         90, "Neutral-warm": 70, "Neutral": 50,
    "Neutral-cool": 35, "Cool / Rosy":  15,
    // Depth
    "Light": 25, "Medium": 55, "Deep": 85,
    // Chroma
    "Muted": 20, "Medium": 50, "Clear / Bright": 85,
    // Contrast
    "Low": 20, "Medium": 55, "High": 90,
  };
  // Camera State & Refs
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleNavigation = (page) => {
    setActivePage(page);
    window.scrollTo(0, 0);
  };

  // --- CAMERA LOGIC ---
  const startCamera = async () => {
    setIsCameraOpen(true);
    try {
      // Requests the front-facing camera
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "user" } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      alert("Camera access denied or unavailable on this device.");
      setIsCameraOpen(false);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsCameraOpen(false);
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert canvas to a File object, exactly like an upload
      canvas.toBlob((blob) => {
        const file = new File([blob], "captured-selfie.png", { type: "image/png" });
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        stopCamera();
      }, 'image/png');
    }
  };

  // Clean up camera if user navigates away while it's open
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // --- UPLOAD LOGIC ---
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const resetPhoto = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setAnalysisData(null);
  };
  const SEASON_DESCRIPTIONS = {
    "Light Spring":   "Delicate and warm, you shine in soft peach, ivory, and golden tones.",
    "True Spring":    "Your skin runs warm with yellow-based undertones, paired with bright, clear coloring.",
    "Bright Spring":  "Vivid and warm — you can carry bold, saturated colors better than any other spring.",
    "Light Summer":   "Soft and cool — dusty rose, lavender, and powder blue are your naturals.",
    "True Summer":    "Cool and muted, you glow in blended, smoky shades and soft pastels.",
    "Soft Summer":    "A gentle blend of cool and neutral — you suit soft, powdery tones.",
    "Soft Autumn":    "Warm and muted, your palette is earthy — think terracotta, moss, and warm taupe.",
    "True Autumn":    "Rich and warm — you carry golden, spicy, and earthy shades effortlessly.",
    "Deep Autumn":    "Deep and warm — chocolate, bronze, and deep olive are your power shades.",
    "True Winter":    "Cool and striking — pure white, black, and icy jewel tones suit you perfectly.",
    "Deep Winter":    "Deep and cool — you command bold, saturated colors like navy, burgundy, and forest.",
    "Bright Winter":  "High contrast and vivid — you glow in sharp, clear, saturated tones.",
  };

  const getSeasonDescription = (season) =>
    SEASON_DESCRIPTIONS[season] ?? "Your personal color has been identified. See your matched products below.";
  
  const handleAnalyze = async () => {
  if (!selectedFile) return;
  setIsAnalyzing(true);

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("https://khoawawa-beauty-backend.hf.space/api/v1/parse-face", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Server error");
    }

    const stat = await response.json();

    // Map backend response → frontend analysisData shape
    const analysisData = {
      analysis: {
        name: stat.personal_color,
        // description: getSeasonDescription(stat.personal_color),
        metrics: {
          undertone: { label: stat.undertone, percentage: METRIC_PERCENTAGES[stat.undertone] ?? 50 },
          depth:     { label: stat.depth,     percentage: METRIC_PERCENTAGES[stat.depth]     ?? 50 },
          chroma:    { label: stat.chroma,    percentage: METRIC_PERCENTAGES[stat.chroma]     ?? 50 },
          contrast:  { label: stat.contrast,  percentage: METRIC_PERCENTAGES[stat.contrast]  ?? 50 },
        },
      },
      recommended_products: [] // wire up later
    };

    setAnalysisData(analysisData);
    handleNavigation('results');
  } catch (error) {
    alert("Analysis failed: " + error.message);
  } finally {
    setIsAnalyzing(false);
  }
};

  const toggleCompare = (product) => {
    const exists = compareList.find((p) => p.id === product.id);
    if (exists) {
      setCompareList(compareList.filter((p) => p.id !== product.id));
    } else {
      if (compareList.length >= 3) {
        alert('You can compare up to 3 products at a time. Remove one first.');
        return;
      }
      setCompareList([...compareList, product]);
    }
  };

  const catNames = { foundation: 'Foundation', blush: 'Blush', eyeshadow: 'Eyeshadow', lipstick: 'Lip Color' };

  const NavItem = ({ page, label }) => (
    <div className={`nav-item ${activePage === page ? 'active' : ''}`} onClick={() => handleNavigation(page)}>
      <span className="dot"></span> {label}
    </div>
  );

  return (
    <div className="app">
      {/* SIDEBAR */}
      <div className="sidebar">
        <div className="logo">AI Beauty <span>Advisor</span></div>
        <div className="tagline">AI analysis, beauty personalization</div>

        <div className="nav-section-title">Explore</div>
        <NavItem page="home" label="Home" />
        <NavItem page="education" label="Learn the Basics" />

        <div className="nav-section-title">Your Analysis</div>
        <NavItem page="upload" label="Upload Selfie" />
        <NavItem page="results" label="Analysis Result" />

        <div className="nav-section-title">Shop &amp; Compare</div>
        <NavItem page="products" label="Recommended Products" />
        <NavItem page="compare" label="Compare Products" />
      </div>

      {/* MAIN CONTENT */}
      <div className="main">

        {/* HOME */}
        {activePage === 'home' && (
          <div className="page active">
            <div className="topbar">
              <div>
                <h1>Welcome back</h1>
                <div className="sub">Discover the colors that were made for you</div>
              </div>
            </div>

            <div className="home-hero">
              <div>
                <h2>Find your <em>personal color</em>, and the makeup that loves it back.</h2>
                <p>Take a selfie or upload a photo, and our AI reads your skin tone, undertone, depth, chroma and contrast — then matches you to shades built for your coloring.</p>
                <button className="btn-primary" onClick={() => handleNavigation('upload')}>Start My Analysis</button>
              </div>
              <div className="hero-circle">Warm Spring<br />could be<br />your match</div>
            </div>

            <div className="grid steps-grid">
              <div className="step-card">
                <div className="step-num">STEP ONE</div>
                <h3>Snap or Upload</h3>
                <p>A clear, front-facing photo in natural light gives the AI the cleanest read.</p>
              </div>
              <div className="step-card">
                <div className="step-num">STEP TWO</div>
                <h3>Get your personal color</h3>
                <p>See your undertone, depth, chroma and contrast laid out simply.</p>
              </div>
              <div className="step-card">
                <div className="step-num">STEP THREE</div>
                <h3>Shop your shades</h3>
                <p>Browse picks matched to your palette, and compare before you buy.</p>
              </div>
            </div>
          </div>
        )}

        {/* UPLOAD / CAMERA PAGE */}
        {activePage === 'upload' && (
          <div className="page active">
            <div className="topbar">
              <div>
                <h1>Provide Your Selfie</h1>
                <div className="sub">Take a live photo or upload one from your device</div>
              </div>
            </div>

            <div className="upload-wrap">
              <div className={`upload-box ${(previewUrl || isCameraOpen) ? 'has-image' : ''}`}>
                
                {/* DEFAULT STATE: Choose Action */}
                {!previewUrl && !isCameraOpen && (
                  <>
                    <div className="upload-icon">📷</div>
                    <h3>How would you like to provide your photo?</h3>
                    <p style={{ marginBottom: '24px' }}>Front-facing, natural light works best</p>
                    
                    <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
                      <button className="btn-primary" onClick={startCamera}>
                        Take a Selfie
                      </button>
                      <button className="btn-outline" onClick={() => fileInputRef.current.click()} style={{ margin: 0 }}>
                        Upload Photo
                      </button>
                    </div>
                  </>
                )}

                {/* CAMERA STATE */}
                {isCameraOpen && (
                  <div className="camera-container">
                    <video ref={videoRef} autoPlay playsInline muted className="live-video"></video>
                    {/* Hidden canvas for capturing the frame */}
                    <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
                    
                    <div style={{ display: 'flex', gap: '12px', justifyContent: 'center', marginTop: '16px' }}>
                      <button className="btn-primary" onClick={capturePhoto}>Snap Photo</button>
                      <button className="btn-outline" onClick={stopCamera} style={{ margin: 0 }}>Cancel</button>
                    </div>
                  </div>
                )}

                {/* PREVIEW STATE */}
                {previewUrl && !isCameraOpen && (
                  <div>
                    <img src={previewUrl} alt="Preview" id="previewImg" />
                    <div style={{ marginTop: '16px' }}>
                      <button className="btn-outline" onClick={resetPhoto} style={{ margin: '0', fontSize: '12px', padding: '8px 16px' }}>
                        Retake / Choose Another
                      </button>
                    </div>
                  </div>
                )}

                <input type="file" ref={fileInputRef} onChange={handleFileChange} style={{ display: 'none' }} accept="image/*" />
              </div>

              {previewUrl && (
                <button className="btn-primary" onClick={handleAnalyze} disabled={isAnalyzing} style={{ marginTop: '24px', width: '100%', maxWidth: '320px' }}>
                  {isAnalyzing ? 'Extracting CIELAB & Running AI...' : 'Analyze My Photo'}
                </button>
              )}
            </div>
          </div>
        )}

        {/* RESULTS */}
        {activePage === 'results' && (
          <div className="page active">
            <div className="topbar">
              <div>
                <h1>Your Analysis Result</h1>
                <div className="sub">Based on the photo you provided</div>
              </div>
            </div>

            {!analysisData ? (
              <div className="compare-empty">
                <p>No analysis data available yet. Please provide a selfie first.</p>
                <button className="btn-primary" onClick={() => handleNavigation('upload')} style={{ marginTop: '20px' }}>Provide Photo</button>
              </div>
            ) : (
              <div className="analysis-layout">
                <div className="result-photo">
                  <img src={previewUrl} alt="Analyzed" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                </div>
                <div className="result-right">
                  <div className="card">
                    <div className="tone-badge">✦ AI Analysis Result</div>
                    <h2>Your personal color is<br /><span className="season">{analysisData.analysis.name}</span></h2>
                    {/* <p>{analysisData.analysis.description}</p> */}

                    <div className="metric-row">
                      {Object.entries(analysisData.analysis.metrics).map(([key, metric]) => (
                        <div className="metric" key={key}>
                          <div className="metric-label">
                            <span style={{ textTransform: 'capitalize' }}>{key} — {metric.label}</span>
                            <span>{metric.percentage}%</span>
                          </div>
                          <div className="bar"><i style={{ width: `${metric.percentage}%` }}></i></div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <button className="btn-primary" onClick={() => handleNavigation('products')} style={{ width: '100%' }}>
                    See My Recommended Products
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* PRODUCTS */}
        {activePage === 'products' && (
          <div className="page active">
            <div className="topbar">
              <div>
                <h1>Recommended For You</h1>
                <div className="sub">Matched to your palette via ΔE optimization</div>
              </div>
            </div>

            <div className="product-tabs">
              {['all', 'foundation', 'lipstick', 'blush', 'eyeshadow'].map((tab) => (
                <div
                  key={tab}
                  className={`product-tab ${productFilter === tab ? 'active' : ''}`}
                  onClick={() => setProductFilter(tab)}
                  style={{ textTransform: 'capitalize' }}
                >
                  {tab === 'lipstick' ? 'Lip Color' : tab}
                </div>
              ))}
            </div>

            <div className="product-grid">
              {analysisData?.recommended_products
                ?.filter((p) => productFilter === 'all' || p.cat === productFilter)
                .map((p) => {
                  const inCompare = compareList.some((item) => item.id === p.id);
                  return (
                    <div className="product-card" key={p.id}>
                      <div className="product-swatch" style={{ background: p.color }}></div>
                      <div className="match-tag">{p.match}% match</div>
                      <div>
                        <div className="brand">{p.brand} · {catNames[p.cat]}</div>
                        <h4>{p.name}</h4>
                        <div style={{ fontSize: '12px', color: 'var(--muted)', marginTop: '2px' }}>Shade: {p.shade}</div>
                      </div>
                      <div className="price">{p.price}</div>
                      <div className="product-actions">
                        <button className={`btn-small ${inCompare ? 'active' : ''}`} onClick={() => toggleCompare(p)}>
                          {inCompare ? 'Added ✓' : 'Compare'}
                        </button>
                      </div>
                    </div>
                  );
                })}
            </div>
            {!analysisData && <p className="compare-empty">Upload a photo to see your matched products.</p>}
          </div>
        )}

        {/* COMPARE */}
        {activePage === 'compare' && (
          <div className="page active">
             <div className="topbar">
              <div>
                <h1>Compare Products</h1>
                <div className="sub">Add up to 3 products from the recommendations page</div>
              </div>
            </div>

            <div className="card">
              {compareList.length === 0 ? (
                <div className="compare-empty">
                  <p>You haven't added any products to compare yet.</p>
                  <button className="btn-primary" onClick={() => handleNavigation('products')} style={{ marginTop: '18px' }}>Browse Products</button>
                </div>
              ) : (
                <div style={{ overflowX: 'auto' }}>
                  <table className="compare-table">
                    <thead>
                      <tr>
                        <th>&nbsp;</th>
                        {compareList.map((p) => (
                          <th key={p.id}>
                            <div className="compare-product-head">
                              <div className="sw" style={{ background: p.color, width: '32px', height: '32px', borderRadius: '8px' }}></div>
                              <div>{p.name}</div>
                              <button className="remove-btn" onClick={() => toggleCompare(p)}>✕</button>
                            </div>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="label">Brand</td>
                        {compareList.map((p) => <td key={`brand-${p.id}`}>{p.brand}</td>)}
                      </tr>
                      <tr>
                        <td className="label">Category</td>
                        {compareList.map((p) => <td key={`cat-${p.id}`}>{catNames[p.cat]}</td>)}
                      </tr>
                      <tr>
                        <td className="label">Shade</td>
                        {compareList.map((p) => <td key={`shade-${p.id}`}>{p.shade}</td>)}
                      </tr>
                      <tr>
                        <td className="label">Price</td>
                        {compareList.map((p) => <td key={`price-${p.id}`}>{p.price}</td>)}
                      </tr>
                      <tr>
                        <td className="label">Match Score</td>
                        {compareList.map((p) => <td key={`match-${p.id}`}>{p.match}% match</td>)}
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}

        {/* EDUCATION */}
        {activePage === 'education' && (
           <div className="page active">
             <div className="topbar">
              <div>
                <h1>Learn the Basics</h1>
                <div className="sub">Simple explanations for the terms behind your results</div>
              </div>
            </div>
            <div className="grid edu-grid">
               <div className="card edu-card">
                 <div className="icon-circle">🌤️</div>
                 <h3>Undertone</h3>
                 <p>Undertone is the subtle color beneath your skin's surface — warm (yellow), cool (pink), or neutral. It is the biggest factor in matching makeup.</p>
               </div>
               <div className="card edu-card">
                 <div className="icon-circle">🌗</div>
                 <h3>Depth</h3>
                 <p>Depth describes how light or dark your overall coloring is. Lighter depths suit softer shades, while deeper depths can carry richer colors.</p>
               </div>
            </div>
           </div>
        )}
      </div>
    </div>
  );
}