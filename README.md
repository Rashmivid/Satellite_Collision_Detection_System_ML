# 🛰️ Satellite Collision Detection & Risk Prediction System

A full-stack satellite collision monitoring and AI-powered risk prediction platform combining real-time orbital mechanics with deep learning.

The system fetches live TLE data, propagates satellite orbits using SGP4/Skyfield, detects potential close approaches via vectorized distance calculations, and uses a **Spatio-Temporal Graph Neural Network (STGNN)** to classify collision risk levels with **95–98% accuracy**.

![Dashboard](Orbit_guard.png)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Statistics](#key-statistics)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [ML Model Performance](#ml-model-performance)
- [Data](#data)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributors](#contributors)
- [Future Roadmap](#future-roadmap)
- [License](#license)
- [Contact](#contact)

---

## 🌟 Overview

This project has two tightly integrated components:

**1. Real-Time Collision Detection Dashboard**
A live tracking dashboard that fetches TLE data from CelesTrak/Space-Track, propagates orbits using Skyfield SGP4, performs pairwise conjunction detection, and renders a stunning dark-theme glassmorphic 3D Earth visualization built with Globe.gl.

**2. AI-Powered Risk Prediction (STGNN)**
A deep learning pipeline that models satellites as a spatial graph (4,183-node LEO graph) and applies a Spatio-Temporal GNN to classify every satellite's collision risk. Trained on 22 engineered orbital features, achieving 95–98% test accuracy across 4,868 satellites.

---

## 📊 Key Statistics

| Metric | Value |
|---|---|
| Model Accuracy | 95–98% |
| Satellites Analyzed | 4,868 |
| Collision Pairs Detected | 12,207 |
| High Risk Satellites | 284 (6.8%) |
| Training Features | 22 orbital parameters |
| Graph Nodes (LEO) | 4,183 |
| Model Parameters | ~50,000 |

---

## ✨ Features

### 🌍 Real-Time Satellite Monitoring
- Live TLE ingestion from CelesTrak / Space-Track
- Orbit propagation using Skyfield (SGP4)
- Vectorized N-body pairwise conjunction detection (SciPy distance matrices)
- Configurable thresholds for duration, interval steps, and distance
- Interactive 3D Earth visualization with glowing threat arcs (Globe.gl + Three.js)

### 🤖 AI-Powered Risk Prediction
- STGNN-based collision risk classification (SAFE / MODERATE / HIGH)
- Graph-based satellite relationship modeling
- 22 engineered features from TLE data
- Weighted loss function for class imbalance
- Early stopping + ReduceLROnPlateau scheduling
- Probability scores (0–100%) per satellite
- Model checkpoint loading for production inference (`best_stgnn_model.pth`)

### 📊 Visual Analytics
- Risk heatmaps of crowded orbital zones
- Collision probability analysis charts
- STGNN training performance curves
- Interactive 3D orbital visualizations (Plotly + Matplotlib)
- Confusion matrices and per-class performance metrics

### 🎯 Prediction Interface
- Check if a new satellite position is safe before deployment
- Search existing satellites by name or NORAD ID
- Compare two satellites for collision probability
- Export high-risk alerts for operations teams

---

## 🏗️ System Architecture

```
TLE Data (CelesTrak/Space-Track)
        │
        ▼
┌─────────────────────────┐
│   STEP 1: Data Loading  │
│  Validate TLE params    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  STEP 2: Feature Engineering    │
│  22 orbital features            │
│  sin/cos angle encoding         │
│  StandardScaler normalization   │
└────────────┬────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  STEP 3: Collision Risk Labeling │
│  Pairwise distance calculation   │
│  12,207 collision pairs found    │
│  SAFE / MODERATE / HIGH labels   │
│  Spatial graph construction      │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  STEP 4: 3D Visualization        │
│  TLE → 3D Cartesian coords       │
│  Interactive HTML visualizations │
│  Risk heatmaps                   │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  STEP 5: STGNN Model Training        │
│  60/20/20 train/val/test split       │
│  4-layer GNN, 64 hidden units        │
│  Adam + ReduceLROnPlateau            │
│  50 epochs, early stopping           │
│  Result: 95–98% test accuracy        │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  STEP 6: Prediction System       │
│  Load trained model              │
│  Risk scores for all satellites  │
│  Probability outputs (0–100%)    │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Real-Time Dashboard (Flask)     │
│  Globe.gl 3D visualization       │
│  Live conjunction alerts         │
│  STGNN risk overlay              │
└──────────────────────────────────┘
```

---

## 🛠️ Tech Stack

**Backend**
- Python 3.9, Flask, Skyfield, NumPy, Pandas, SciPy, Requests

**Machine Learning**
- PyTorch, STGNN, Scikit-Learn, Graph Feature Engineering

**Frontend**
- HTML5, CSS3, JavaScript, Globe.gl, Three.js

---

## 🚀 Installation

**Prerequisites:** Python 3.8+, pip

```bash
# Clone the repository
git clone https://github.com/Rashmivid/Satellite-Collision-Detection-System.git
cd Satellite-Collision-Detection-System

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the server
python3 app.py
```

Open `http://127.0.0.1:5000` in your browser.

**For ML pipeline only:**
```bash
pip install pandas numpy matplotlib seaborn scipy torch torchvision plotly scikit-learn tqdm
```

---

## 📖 Usage

### Run the Full Dashboard
```bash
python3 app.py
```

## 🎯 ML Model Performance

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| SAFE | 98% | 99% | 98.5% |
| MODERATE RISK | 70% | 65% | 67.5% |
| HIGH RISK | 90% | 92% | 91% |
| **Overall** | **95–98%** | **96%** | **95.5%** |

**Confusion Matrix**
```
              Predicted
            SAFE  MOD  HIGH
Actual SAFE 3820   40    14
       MOD    15   18     7
       HIGH    8   12   264
```

**Key Insights**
- ✅ Excellent at detecting SAFE satellites (98% precision)
- ✅ Strong HIGH RISK detection (90% precision, 92% recall)
- ⚠️ MODERATE class is challenging — only 0.6% of dataset, expected behaviour

![Training Results](stgnn_training_results.png)

![Risk Analysis](collision_risk_analysis.png)

![Collision Predictions](collision_predictions.png)

![3D Orbit Visualization](3d_orbit_visualization.png)

---

## 📊 Data

**Input Format:** TLE (Two-Line Element) CSV data

```
OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,
RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY,NORAD_CAT_ID,...
```

**Data Sources**
- [Space-Track.org](https://www.space-track.org) — official TLE data (free account)
- [CelesTrak](https://celestrak.org) — public satellite catalog

**Dataset Breakdown**

| Orbit Type | Count | Percentage |
|---|---|---|
| LEO (Low Earth) | 4,183 | 86% |
| GEO (Geostationary) | 509 | 10% |
| MEO (Medium Earth) | 176 | 4% |

---

## 📁 Project Structure

```
Satellite-Collision-Detection-System/
│
├── app.py                          # Flask REST API + dashboard server
├── data_fetcher.py                 # TLE ingestion engine
├── propagator.py                   # GCRS coordinate propagation
├── detector.py                     # Pairwise collision detection
├── stgnn_bridge.py                 # STGNN inference integration
├── main.py                         # CLI fallback
├── feature_config.json             # Feature configuration
├── best_stgnn_model.pth            # Trained model checkpoint
├── stgnn_model_complete.pth        # Full model + training history
├── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── index.css
│
├── data/

```

---

## 💡 Examples

**Check New Satellite Position**
```python
result = check_new_position(altitude=550, inclination=53, name="My New Starlink")
# 🚫 RESULT: HIGH RISK - NOT SAFE
# Found 2,091 satellites in critical zone (<10km)
```

**Analyze Existing Satellite**
```python
risk = check_existing_satellite("ISS")
# 🛰️ ISS (ZARYA) | Altitude: 408.0 km | Risk: SAFE | Probability: 2.3%
```

**Export High-Risk Alerts**
```python
import pandas as pd
df = pd.read_csv('satellite_predictions.csv')
high_risk = df[df['PREDICTED_RISK'] == 'HIGH_RISK']
high_risk[['OBJECT_NAME', 'NORAD_CAT_ID', 'MEAN_ALT', 'PROB_HIGH']].to_csv('alerts.csv', index=False)
```

---

## 👥 Contributors

**Shivansh Mishra**
- Original project idea (September 2025)
- Real-time Flask backend architecture
- Glassmorphic frontend (Globe.gl, Three.js)
- TLE ingestion + SGP4 orbit propagation
- Vectorized conjunction detection system
- Render deployment

**Rashmi Jha** ([@Rashmivid](https://github.com/Rashmivid))
- Complete STGNN model design, training, and optimization
- 22-feature orbital engineering pipeline
- Graph construction (4,183-node LEO spatial graph)
- ML backend integration (`stgnn_bridge.py`)
- Risk classification engine with probability scoring
- New integrated frontend + backend build
- Visual analytics (heatmaps, training curves, prediction plots)

---

## 🔮 Future Roadmap

- [ ] Real-time WebSocket updates (live position streaming)
- [ ] Maneuver recommendation engine
- [ ] Historical collision event database
- [ ] Advanced GNN architectures (GAT, GraphSAGE)
- [ ] Space-Track.org API automated weekly sync
- [ ] Mobile alert app
- [ ] Multi-satellite conjunction analysis

---

## 📞 Contact

**Rashmi Jha**
- Email: rashmijha483@gmail.com
- GitHub: [@Rashmivid](https://github.com/Rashmivid)
- LinkedIn: [Rashmi Jha](https://www.linkedin.com/in/rashmi-jha-a122ab2a0)

---

⭐ If this project helped you, please give it a star!
