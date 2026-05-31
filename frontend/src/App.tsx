import { useMemo, useState, type DragEvent } from 'react';

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

type DetectionBox = {
  class_id: number;
  confidence: number;
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
};

type ResultPayload = {
  disease: string;
  confidence: number;
  bounding_boxes: DetectionBox[];
  model_used: string;
  inference_time: string;
  uploaded_image: string;
};

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<ResultPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');
  const [error, setError] = useState<string | null>(null);

  const themeClass = theme === 'dark' ? 'theme-dark' : 'theme-light';

  const handleFileChange = (file: File) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) handleFileChange(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('image_file', selectedFile);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Prediction error');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleCapture = async () => {
    setLoading(true);
    setError(null);
    setSelectedFile(null);
    setPreviewUrl(null);

    try {
      const response = await fetch(`${API_URL}/capture`);
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Camera capture failed');
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const toggleTheme = () => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));

  const formattedBoxes = useMemo(() => {
    return result?.bounding_boxes.map((box, index) => (
      <li key={index}>
        Class {box.class_id} ({box.confidence.toFixed(2)}) — [{box.xmin.toFixed(0)}, {box.ymin.toFixed(0)}] → [{box.xmax.toFixed(0)}, {box.ymax.toFixed(0)}]
      </li>
    ));
  }, [result]);

  return (
    <div className={`app-shell ${themeClass}`}>
      <header className="hero-panel">
        <div>
          <p className="eyebrow">Lemon Leaf Disease Detection</p>
          <h1>Smart plant health monitoring with AI explainability</h1>
          <p className="hero-copy">
            Upload a leaf image, detect disease with DenseNet121, and localize symptoms with YOLOv10.
          </p>
        </div>
        <button className="theme-toggle" onClick={toggleTheme}>
          {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
        </button>
      </header>

      <section className="glass-card intro-card">
        <h2>About the model</h2>
        <div className="grid two-column">
          <article>
            <h3>Classification</h3>
            <p>DenseNet121-based disease classifier trained to identify common lemon leaf conditions with strong accuracy.</p>
          </article>
          <article>
            <h3>Localization</h3>
            <p>YOLOv10 locates affected regions on lemon leaf imagery so you can see where symptoms appear.</p>
          </article>
        </div>
      </section>

      <section className="glass-card upload-card">
        <div className="upload-header">
          <h2>Upload leaf image</h2>
          <p>Drag and drop or choose a file to predict disease.</p>
        </div>

        <div
          className="dropzone"
          onDrop={handleDrop}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => event.preventDefault()}
        >
          {previewUrl ? (
            <img src={previewUrl} alt="Preview" className="preview-image" />
          ) : (
            <div>
              <strong>Drag & drop</strong>
              <span>PNG, JPG, JPEG, WEBP</span>
            </div>
          )}
          <input
            type="file"
            accept="image/*"
            className="file-input"
            onChange={(event) => {
              if (event.target.files?.[0]) handleFileChange(event.target.files[0]);
            }}
          />
        </div>

        <div className="button-group">
          <button className="primary-button" disabled={!selectedFile || loading} onClick={handleUpload}>
            {loading ? 'Detecting...' : 'Detect Disease'}
          </button>
          <button className="secondary-button" disabled={loading} onClick={handleCapture}>
            {loading ? 'Capturing...' : 'Capture from Camera'}
          </button>
        </div>
        {error && <p className="error-text">{error}</p>}
      </section>

      {result && (
        <section className="glass-card result-card">
          <div className="result-grid two-column">
            <div className="result-panel">
              <h2>Results Dashboard</h2>
              <p className="result-pill">Model: {result.model_used}</p>
              <div className="result-stats">
                <div>
                  <span>Prediction</span>
                  <strong>{result.disease}</strong>
                </div>
                <div>
                  <span>Confidence</span>
                  <strong>{(result.confidence * 100).toFixed(2)}%</strong>
                </div>
                <div>
                  <span>Inference Time</span>
                  <strong>{result.inference_time}</strong>
                </div>
              </div>
              <div className="box-list">
                <h3>Detected bounding boxes</h3>
                <ul>{formattedBoxes}</ul>
              </div>
            </div>

            <div className="preview-panel">
              <div className="preview-block">
                <h3>Uploaded image</h3>
                <img src={`${API_URL}${result.uploaded_image}`} alt="Uploaded leaf" />
              </div>
              </div>
          </div>
        </section>
      )}
    </div>
  );
}

export default App;
