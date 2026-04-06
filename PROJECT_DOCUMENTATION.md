# Landslide Susceptibility Prediction System - Complete Project Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Complete Data Pipeline](#complete-data-pipeline)
4. [Model Training & Validation](#model-training--validation)
5. [Susceptibility Map Generation](#susceptibility-map-generation)
6. [Backend API Structure](#backend-api-structure)
7. [Frontend Architecture](#frontend-architecture)
8. [Running the Project](#running-the-project)
9. [File Structure & Descriptions](#file-structure--descriptions)
10. [Key Technologies & Dependencies](#key-technologies--dependencies)
11. [Model Performance Metrics](#model-performance-metrics)

---

## Project Overview

### What is This Project?

This is a **Machine Learning-based Landslide Susceptibility Prediction System** for the Sindhupalchowk district in Nepal. The system predicts the likelihood of landslides occurring at any location within the study area using geospatial data and a trained Random Forest classifier.

### Problem Statement

The Sindhupalchowk district is in the Himalayan region, which is highly prone to landslides. Traditional landslide hazard assessment is time-consuming and requires expert knowledge. This system automates the process by:

- Training a machine learning model on historical landslide locations and geomorphological features
- Generating a continuous probability map showing landslide susceptibility across the entire region
- Providing an interactive web interface for visualizing risk zones
- Allowing users to click on any location to get a real-time prediction

### Key Objectives

1. **Data Integration**: Combine multiple geospatial datasets (rasters and vector files) into a unified analysis framework
2. **Feature Engineering**: Extract meaningful geomorphological features from raw data (slope, aspect, elevation, drainage, etc.)
3. **Model Development**: Train a Random Forest classifier that achieves ~82% accuracy
4. **Prediction**: Apply the model across the entire study area to create a susceptibility map
5. **Visualization**: Provide an interactive web interface for exploring predictions

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     WEB BROWSER                             │
│               (React Frontend)                              │
└────────┬────────────────────────────────────────────────────┘
         │ HTTP Requests (JSON)
         │
┌────────▼────────────────────────────────────────────────────┐
│              FASTAPI BACKEND SERVER                         │
│         (uvicorn on http://localhost:8000)                  │
│                                                             │
│  • Serves ML predictions                                   │
│  • Returns susceptibility map as image tiles               │
│  • Provides landslide locations (GeoJSON)                  │
│  • Serves pre-computed statistics                          │
└────────┬────────────────────────────────────────────────────┘
         │ File System Access
         │
┌────────▼────────────────────────────────────────────────────┐
│         PRE-COMPUTED DATA & MODEL FILES                     │
│                                                             │
│  • model.joblib (Random Forest)                            │
│  • susceptibility_map.tif (GeoTIFF raster)                 │
│  • training_data.csv (training dataset)                    │
│  • feature_metadata.json (metadata)                        │
│  • Geospatial rasters (DEM, slope, aspect, etc.)           │
│  • Vector data (landslides, rivers, soil, landuse)         │
└─────────────────────────────────────────────────────────────┘
```

### Separation of Concerns

- **Frontend**: React SPA that provides interactive map visualization and UI
- **Backend**: Python FastAPI REST API that serves predictions and data
- **Data Layer**: Pre-computed rasters, vector data, and trained model
- **Processing**: Happens offline (data prep, model training) before deployment

---

## Complete Data Pipeline

### Overview

The data pipeline consists of three main stages:

1. **Data Preparation** (prepare_data.py)
2. **Model Training** (data.py)
3. **Susceptibility Map Generation** (generate_susceptibility_map.py)

### Stage 1: Data Preparation (`prepare_data.py`)

#### Purpose

Transform raw geospatial data into a machine learning-ready CSV dataset by:

- Harmonizing different data sources into a common coordinate system
- Sampling balanced landslide/non-landslide points
- Extracting feature values at each sample point

#### Input Data

**Raster Files** (GeoTIFF format):

```
Dem_sind.tif                  → Digital Elevation Model (m)
aspect.tif                    → Slope aspect (degrees)
slope_modified.tif            → Slope gradient (degrees)
Slope_main.tif                → Primary slope direction
drainage_direction.tif        → Flow direction
hillshade2.tif                → Hillshade for visualization
twi.tif                       → Topographic Wetness Index
```

**Vector Files** (GeoPackage format):

```
landslide.gpkg                → Known historical landslides (polygon/point)
soil Parent.gpkg              → Soil type classifications (polygon)
river line.gpkg               → River networks (line)
river polygon.gpkg            → Flood-prone areas (polygon)
sind_land_use.gpkg            → Land use/land cover classifications (polygon)
Sindhupalchowk_boundry.gpkg   → Study area boundary (polygon)
```

#### Process Steps

1. **Load & Validate Data**
   - Open all rasters and vector files
   - Check coordinate systems and spatial extents
   - Identify and handle data quality issues

2. **Reproject to Common CRS**
   - Target CRS: **EPSG:32645** (UTM Zone 45N)
   - All rasters and vectors standardized to this system
   - Ensures pixel-perfect spatial alignment

3. **Generate Training Samples**

   ```
   Positive samples (landslides):    380 points from known landslide areas
   Negative samples (non-landslides): 380 points from stable areas
   Total samples:                    760 training points
   ```

4. **Extract Features at Sample Points**

   For each of the 760 sample points, the system extracts:
   - **DEM (dem)**: Elevation value at that location
   - **Aspect**: Slope direction (N, S, E, W, etc.) → categorical
   - **Slope**: Steepness of terrain → continuous
   - **Slope Main**: Primary flow direction slope
   - **Drainage**: Flow accumulation from watershed analysis
   - **Hillshade**: Visual relief (derived from DEM)
   - **TWI**: Topographic Wetness Index (combines slope + accumulated flow)
   - **Distance to River (dist_river)**: Kilometers to nearest river (computed)
   - **Soil Type (soil_type)**: Categorical soil classification (extracted from vector data)
   - **Land Use (landuse)**: Categorical land cover type (extracted from vector data)

   **Computed Features**:
   - Distance to River: Rasterized river lines → Euclidean distance transform → scale to meters
   - Soil Type: Rasterize soil polygons → assign category codes (1-8)
   - Land Use: Rasterize land use polygons → assign category codes

5. **Output**
   ```
   training_data.csv              → 760 rows × 12 columns (10 features + x/y coordinates + label)
   feature_metadata.json          → Metadata about data process (CRS, raster bounds, feature names)
   ```

#### Key Configuration

```python
N_SAMPLES_PER_CLASS = 2000  # In the script, but only 380 per class in actual dataset
TARGET_CRS = "EPSG:32645"   # UTM Zone 45N
```

---

### Stage 2: Model Training (`data.py`)

#### Purpose

Train a Random Forest classifier on the prepared CSV dataset, validate its performance, and save the trained model and metrics.

#### Input

- `training_data.csv` (760 samples, 10 features)
- Feature values and binary landslide labels (1 = landslide, 0 = stable)

#### Model Architecture

**Algorithm**: Random Forest Classifier
**Hyperparameters**:

- `n_estimators`: 200 trees
- `max_depth`: 10 levels per tree
- Other parameters: sklearn defaults

**Rationale**:

- Random Forests are robust to outliers in geospatial data
- Handle mixed feature types (continuous + categorical)
- Provide feature importance rankings
- Computationally efficient for real-time prediction

#### Training Process

1. **Data Split**
   - Training set: 608 samples (80%)
   - Test set: 152 samples (20%)
   - Stratified split to maintain class balance

2. **Model Training**
   - Fit 200 decision trees on training data each on random subsets of features and samples
   - Random Forest averages predictions across all trees

3. **Model Evaluation**
   - Test on held-out test set
   - Evaluate classification metrics

4. **Cross-Validation**
   - 5-fold cross-validation on training data
   - Measure variability across folds

5. **Feature Importance**
   - Extract importance scores from Random Forest
   - Rank features by their contribution to predictions

#### Output Files

```
model.joblib                   → Serialized trained Random Forest (joblib format)
model_metrics.json             → Performance metrics and feature importance
```

---

### Stage 3: Susceptibility Map Generation (`generate_susceptibility_map.py`)

#### Purpose

Apply the trained model across the entire raster extent to create a continuous susceptibility probability map showing landslide likelihood at every pixel.

#### Process

1. **Load Resources**
   - Load trained model from model.joblib
   - Load feature metadata
   - Load reference raster (slope_modified.tif) as spatial reference

2. **Prepare Full-Coverage Feature Grids**

   For each raster and vector-derived feature:
   - Load or compute the feature layer spanning the entire study area
   - Reproject to match reference raster if needed
   - Handle nodata values (replace with NaN)

   Example: For slope feature, every pixel in slope_modified.tif becomes a feature input

3. **Stack Features into 3D Array**

   ```
   Shape: (n_features, height, width) = (10, 2281, 2134)

   Each pixel position has a vector of 10 feature values:
   [dem, aspect, slope, slope_main, drainage, hillshade, twi,
    dist_river, soil_type, landuse]
   ```

4. **Apply Model**

   ```python
   # Reshape to (n_pixels, n_features) = (4,863,854, 10)
   X = features.reshape(n_features, -1).T

   # Predict probability of landslide at each pixel
   probabilities = model.predict_proba(X)[:, 1]  # Get probability of class 1

   # Reshape back to 2D map (height, width)
   susceptibility_map = probabilities.reshape((height, width))
   ```

5. **Write Output GeoTIFF**
   - Save as susceptibility_map.tif with proper geospatial metadata
   - CRS: EPSG:32645
   - Transform: matches reference raster
   - Values: 0.0 - 1.0 (probability of landslide)

#### Output Files

```
susceptibility_map.tif         → Full raster map of susceptibility (0-1)
susceptibility_zones.tif       → Discretized vulnerability zones (5 classes)
```

#### Map Interpretation

- **Dark red/high values (0.8-1.0)**: Very high susceptibility
- **Orange/medium-high (0.6-0.8)**: High susceptibility
- **Yellow/medium (0.4-0.6)**: Moderate susceptibility
- **Light green/medium-low (0.2-0.4)**: Low susceptibility
- **Dark green/low values (0.0-0.2)**: Very low susceptibility

---

## Model Training & Validation

### Model Performance Metrics

```json
{
  "accuracy": 0.8224, // Overall correct predictions
  "precision": 0.8101, // Of predicted landslides, % correct
  "recall": 0.8421, // Of actual landslides, % found
  "f1_score": 0.8258, // Harmonic mean of precision/recall
  "roc_auc": 0.8826, // Area under ROC curve (discrimination ability)
  "cv_auc_mean": 0.8476, // Cross-validation AUC (5-fold)
  "cv_auc_std": 0.0304 // Variability across CV folds
}
```

### Performance Interpretation

- **Accuracy (82.2%)**: Of all predictions, 82% are correct
- **Precision (81%)**: When model says "landslide", it's correct 81% of the time → low false alarms
- **Recall (84%)**: Model catches 84% of actual landslides → good coverage
- **ROC-AUC (0.883)**: Model has excellent discrimination between classes (>0.8 is excellent)
- **Consistency**: Low CV AUC std dev (0.03) means model generalizes well across data subsets

### Confusion Matrix

```
                Predicted Stable    Predicted Landslide
Actual Stable        61 (TN)              15 (FP)
Actual Landslide     12 (FN)              64 (TP)
```

- **True Negatives (61)**: Correctly identified non-landslides
- **True Positives (64)**: Correctly identified landslides
- **False Positives (15)**: Incorrectly flagged as landslide (acceptable - conservative)
- **False Negatives (12)**: Missed landslides (undesirable, but rate is acceptable at 16%)

### Feature Importance Ranking

The model's decision-making is driven primarily by these features (% contribution):

1. **Hillshade (19.9%)** → Visual relief details matter most
2. **Slope (13.4%)** → Steeper terrain = higher risk
3. **DEM (13.3%)** → Elevation influences landslide
4. **Slope Main (12.0%)** → Primary flow direction relevant
5. **Aspect (10.8%)** → Slope direction (N/S exposure) matters
6. **Drainage (8.4%)** → Flow accumulation indicates water concentration
7. **Distance to River (7.6%)** → Proximity to water features relevant
8. **TWI (5.7%)** → Wetness index less important than slope
9. **Soil Type (5.7%)** → Soil properties moderately important
10. **Land Use (3.4%)** → Vegetation cover least important

---

## Susceptibility Map Generation

### Output Map Characteristics

**File**: `susceptibility_map.tif`

- **Format**: GeoTIFF (single-band raster)
- **Width**: 2,134 pixels
- **Height**: 2,281 pixels
- **Pixel Size**: ~29 meters
- **CRS**: EPSG:32645 (UTM Zone 45N)
- **Values**: 0.0 (no landslide) to 1.0 (certain landslide)
- **Data Type**: Float32

### Map Statistics

Across 4,863,854 pixels in the study area:

- **Mean susceptibility**: ~0.45
- **Areas of concern** (susceptibility > 0.6): Concentrated in steep slopes and river valleys
- **Safe zones** (susceptibility < 0.3): High-altitude plateaus, gentle slopes

### Visualization Strategy

In the web interface:

- Map is displayed as a semi-transparent colored overlay on satellite imagery
- Colors transition from green (safe) to red (high risk)
- User can toggle visibility of different overlays

---

## Backend API Structure

### Setup

**Server**: FastAPI (async Python web framework)
**Address**: http://localhost:8000
**Port**: 8000

### Architecture Overview

```
FastAPI Application
├── Load & Cache Resources at Startup
│   ├── model.joblib
│   ├── feature_metadata.json
│   ├── model_metrics.json
│   ├── Raster datasets (all 10 features)
│   ├── susceptibility_map.tif
│   └── landslide.gpkg
│
├── CORS Middleware (allow frontend requests)
├── Static Files (serve compiled React app from frontend/dist/)
│
└── API Endpoints
    ├── GET /api/susceptibility-image      → Susceptibility map as PNG
    ├── GET /api/landslides                → Historical landslides as GeoJSON
    ├── GET /api/stats                     → Global statistics
    ├── GET /api/model-metrics             → Model performance details
    ├── GET /api/map-center                → Study area center coordinates
    └── POST /api/predict                  → Point-level prediction
```

### Startup Procedure (`main.py`)

When the FastAPI server starts:

1. **Initialize Paths**

   ```python
   BASE_DIR = project root
   MODEL_PATH = BASE_DIR / "model.joblib"
   SUSC_MAP_PATH = BASE_DIR / "susceptibility_map.tif"
   # ... etc for all resource files
   ```

2. **Load Model**

   ```python
   model = joblib.load(MODEL_PATH)  # Load trained Random Forest
   ```

3. **Load Metadata**

   ```python
   with open(feature_metadata.json) as f:
       feature_meta = json.load(f)

   feature_names = feature_meta["features"]      # [dem, aspect, slope, ...]
   raster_files = feature_meta["raster_files"]   # Maps feature names to filenames
   ```

4. **Load Susceptibility Map**

   ```python
   susc_ds = rasterio.open(SUSC_MAP_PATH)
   susc_data = susc_ds.read(1)          # Read raster band 1 into memory
   ```

5. **Load Feature Rasters**
   - Open all 10 feature rasters (DEM, slope, aspect, etc.)
   - Load into memory as NumPy arrays
   - Handle CRS/projection issues (reproject if needed)
   - Handle nodata values (convert to NaN)

   This enables fast pixel lookups during point predictions

6. **Compute Derived Features**
   - River distance: Rasterize river lines → distance transform
   - Soil type: Rasterize soil polygons → category codes
   - Land use: Rasterize land use polygons → category codes

---

### Key API Endpoints

#### 1. **POST /api/predict**

**Purpose**: Get landslide prediction at a specific (X, Y) coordinate

**Request**:

```json
{
  "x": 370000.0,
  "y": 3080000.0
}
```

**Response**:

```json
{
  "probability": 0.734,
  "category": "High",
  "feature_values": {
    "dem": 2150.5,
    "aspect": 145.2,
    "slope": 38.4,
    "slope_main": 42.1,
    "drainage": 234.8,
    "hillshade": 189.2,
    "twi": 7.3,
    "dist_river": 450.0,
    "soil_type": 3,
    "landuse": 2
  }
}
```

**Process**:

1.  Convert (X, Y) to pixel row/col using raster transform
2.  Extract 10 feature values at that pixel from loaded raster arrays
3.  Create feature vector
4.  Call `model.predict_proba(feature_vector)` → get probability
5.  Map probability to risk category
6.  Return all info

#### 2. **GET /api/susceptibility-image**

**Purpose**: Get the entire susceptibility map as a PNG image for map display

**Response**: PNG image (compressed, suitable for web)

**Process**:

1.  Load susceptibility_map.tif from disk (or cache)
2.  Normalize values to 0-255 for PNG
3.  Apply color ramp (green→yellow→orange→red)
4.  Add transparency (semi-opaque for overlay)
5.  Return as PNG bytes

#### 3. **GET /api/landslides**

**Purpose**: Get historical landslide locations as GeoJSON

**Response**:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Point", "coordinates": [369500, 3085200] },
      "properties": { "id": 1 }
    },
    ...
  ]
}
```

**Uses**: landslide.gpkg (read with geopandas, convert to GeoJSON)

#### 4. **GET /api/stats**

**Purpose**: Get global statistics for the study area

**Response**:

```json
{
  "mean_susceptibility": 0.452,
  "high_susceptibility_area": 0.23,
  "low_susceptibility_area": 0.67,
  "area_km2": 2800.0
}
```

#### 5. **GET /api/model-metrics**

**Purpose**: Get model performance information

**Response**: Contents of model_metrics.json with accuracy, precision, AUC, feature importance, etc.

#### 6. **GET / (Static Files)**

**Purpose**: Serve the React frontend (index.html, JS bundles, CSS)

**Source**: `frontend/dist/` (output of Vite build)

---

## Frontend Architecture

### Technology Stack

- **Framework**: React 19
- **Build Tool**: Vite (fast bundler)
- **Map Library**: Leaflet 1.9.4
- **Chart Library**: Chart.js 4.5
- **Styling**: Custom CSS

### Project Structure

```
frontend/
├── src/
│   ├── App.jsx                    # Main app component
│   ├── main.jsx                   # React entry point
│   ├── components/
│   │   ├── MapCanvas.jsx          # Interactive map display
│   │   ├── PredictionPanel.jsx    # Point prediction interface
│   │   ├── StatsPanel.jsx         # Global statistics
│   │   └── LayersPanel.jsx        # Toggle overlays (susceptibility, landslides, etc.)
│   └── main.css
├── index.html                     # HTML template
├── package.json                   # Dependencies
└── vite.config.js                 # Build configuration
```

### Component Breakdown

#### **App.jsx** (Main Component)

**Responsibilities**:

- State management for entire app (Rect hooks)
- Fetch initial data from backend (susceptibility, landslides, stats)
- Render layout with map and control panels
- Handle panel open/close toggles
- Coordinate data between child components

**Key State**:

```javascript
const [susceptibilityData, setSusceptibilityData]; // Raster image
const [landslidesGeojson, setLandslidesGeojson]; // GeoJSON points
const [stats, setStats]; // Global stats
const [prediction, setPrediction]; // Result of point prediction
const [layerToggles, setLayerToggles]; // Which overlays visible
```

**Startup Flow**:

1. Component mounts → useEffect runs
2. Fetch data in parallel:
   - `/api/susceptibility-image` → convert to base64 for display
   - `/api/landslides` → store GeoJSON
   - `/api/stats` → display summary
   - `/api/nepal-boundary` → display country border
   - `/api/sindhupalchowk-boundary` → highlight study area
3. Add loading overlay while fetching
4. Fade out loading screen once data ready

#### **MapCanvas.jsx** (Interactive Map)

**Responsibilities**:

- Render Leaflet map with tiles
- Display susceptibility map overlay
- Show landslide locations
- Handle click events to trigger predictions
- Update map based on layer toggles

**Features**:

```javascript
// Tile layers
L.tileLayer("https://{s}.tile.openstreetmap.org/...")  // Base map
+ Custom WMS/image layer for susceptibility_map.tif

// Vector overlays
GeoJSON layer for historical landslides (red circles)
GeoJSON layer for study area boundary (blue outline)

// Interaction
onClick → get coordinates → send to backend /api/predict
```

**Map Center**: Sindhupalchowk district (latitude ~28, longitude ~85)

#### **PredictionPanel.jsx** (Point-Level Prediction)

**Responsibilities**:

- Display results after user clicks map
- Show probability as percentage
- Show categorical risk level (Low/Medium/High/Very High)
- List extracted feature values
- Show visual indicator (progress bar or color-coded risk)

**Data Display**:

```
Click Location: (369500 m E, 3085200 m N)
Landslide Probability: 73.4%
Risk Category: HIGH ⚠️

Feature Values Extracted:
├─ Elevation (DEM): 2150.5 m
├─ Slope: 38.4°
├─ Aspect: SE (145°)
├─ Distance to River: 450 m
└─ ... (8 more features)
```

#### **StatsPanel.jsx** (Global Statistics)

**Responsibilities**:

- Display summary statistics from `/api/stats`
- Show charts of spatial distribution
- Summarize model performance

**Content**:

- Mean susceptibility across region
- % area in each risk category
- Area calculations (km²)
- Chart of susceptibility distribution

#### **LayersPanel.jsx** (Layer Controls)

**Responsibilities**:

- Toggle visibility of overlay layers
- Control transparency of susceptibility map

**Layer Toggles**:

```
☑ Susceptibility Map       (semi-transparent colored overlay)
☑ Historical Landslides    (red circle markers)
☑ Satellite Imagery        (high-res background tiles)
☑ Study Area Boundary      (blue polygon outline)
☑ Nepal Border             (country boundary)
```

---

## Running the Project

### Prerequisites

- Python 3.8+ (with venv)
- Node.js 14+ (for frontend)
- Git

### Backend Setup

1. **Create Virtual Environment**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows PowerShell
   ```

2. **Install Dependencies**

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Run Server**

   ```bash
   uvicorn main:app --port 8000 --reload
   ```

   Output:

   ```
   INFO:     Uvicorn running on http://127.0.0.1:8000
   ```

### Frontend Setup

1. **Install Dependencies**

   ```bash
   cd frontend
   npm install
   ```

2. **Run Development Server**

   ```bash
   npm run dev
   ```

   Output:

   ```
   VITE v8.0.2  ready in XXX ms

   ➜  Local:   http://localhost:5173/
   ```

3. **Access Application**
   - Open browser → http://localhost:5173
   - Map loads with susceptibility overlay
   - Click locations to get predictions

### Production Build

1. **Build Frontend**

   ```bash
   cd frontend
   npm run build
   ```

   Output: optimized files in `frontend/dist/`

2. **Backend serves frontend**

   ```bash
   # Backend automatically serves dist/ folder at /
   # No separate frontend server needed in production
   uvicorn main:app --port 8000
   ```

3. **Access**
   - http://localhost:8000 (backend serves frontend + API)

---

## File Structure & Descriptions

### Root Directory

```
Landslide/
├── 📄 readme.md                        # Basic startup instructions
├── 📄 PROJECT_DOCUMENTATION.md         # This file
├── 📄 validation.ipynb                 # Jupyter notebook for data validation
│
├── 🔧 Data Files (Pre-computed)
│   ├── training_data.csv               # 760 samples, 10 features + labels
│   ├── model.joblib                    # Trained Random Forest classifier
│   ├── model_metrics.json              # Performance metrics & feature importance
│   ├── feature_metadata.json           # Metadata about training process
│   ├── susceptibility_map.tif          # Full raster prediction map
│   ├── susceptibility_zones.tif        # Discretized zones (5 classes)
│   ├── landslide.gpkg                  # Historical landslide database
│   └── convert_out.txt                 # Output from data conversion scripts
│
├── 📊 Raster Data (GeoTIFF)
│   ├── Dem_sind.tif                    # Digital Elevation Model (2281×2134 pixels)
│   ├── aspect.tif                      # Slope aspect/direction
│   ├── slope_modified.tif              # Slope gradient (reference raster)
│   ├── Slope_main.tif                  # Primary slope direction
│   ├── drainage_direction.tif          # Flow accumulation
│   ├── hillshade2.tif                  # Hillshade visualization (1.9 GB) [NOT TRACKED]
│   ├── twi.tif                         # Topographic Wetness Index
│   └── open_street.tif                 # Openstreetmap data
│
├── 📍 Vector Data (GeoPackage)
│   ├── Sindhupalchowk_boundry.gpkg     # Study area boundary
│   ├── sind_land_use.gpkg              # Land use classifications
│   ├── soil Parent.gpkg                # Soil type polygons
│   ├── river line.gpkg                 # River centerlines
│   ├── river polygon.gpkg              # River/flood polygons
│   ├── Nepal.shp/Nepal.shx             # Nepal country boundary
│   └── .gitignore                      # Ignore hillshade2.tif (too large)
│
├── 📝 Python Scripts
│   ├── prepare_data.py                 # Stage 1: Data preparation & sampling
│   ├── prepare_data_landuse.py         # Additional land use processing
│   ├── data.py                         # Stage 2: Model training
│   ├── generate_susceptibility_map.py  # Stage 3: Map generation
│   ├── inspect_data.py                 # Data quality inspection
│   └── [.venv/]                        # Python virtual environment (local)
│
├── 🔙 backend/ (FastAPI Server)
│   ├── main.py                         # FastAPI application
│   ├── requirements.txt                # Python dependencies
│   ├── __pycache__/                    # Compiled Python files
│   ├── nepal.geojson                   # Nepal boundary as GeoJSON
│   └── sindhupalchowk.geojson          # Study area as GeoJSON
│
└── 🎨 frontend/ (React App)
    ├── package.json                    # NPM dependencies
    ├── vite.config.js                  # Vite build config
    ├── index.html                      # HTML entry point
    ├── src/
    │   ├── main.jsx                    # React entry point
    │   ├── App.jsx                     # Main app component
    │   ├── main.css                    # Global styles
    │   └── components/
    │       ├── MapCanvas.jsx           # Map rendering component
    │       ├── PredictionPanel.jsx     # Prediction results display
    │       ├── StatsPanel.jsx          # Statistics display
    │       └── LayersPanel.jsx         # Layer control component
    ├── [node_modules/]                 # NPM packages (local, excluded from git)
    ├── [dist/]                         # Built output (created by npm run build)
    └── package-lock.json               # Dependency lock file
```

### Key Data Files Explained

#### `training_data.csv`

```
Format: CSV with header
Columns: dem, aspect, slope, slope_main, drainage, hillshade, twi, dist_river, soil_type, landuse, x, y, label
Rows: 760 (380 landslides + 380 non-landslides)
Size: ~60 KB

Example row:
2145.3,125.4,38.2,41.5,234.2,189.1,7.2,450.5,3,2,369500.2,3085200.5,1

Where:
- Values 0-9: feature values
- x, y: UTM coordinates (EPSG:32645)
- label: 1=landslide, 0=non-landslide
```

#### `model_metrics.json`

```
Contains:
{
  "accuracy": 0.8224,
  "precision": 0.8101,
  "recall": 0.8421,
  "f1_score": 0.8258,
  "roc_auc": 0.8826,
  "cv_auc_mean": 0.8476,
  "cv_auc_std": 0.0304,
  "confusion_matrix": { "TN": 61, "FP": 15, "FN": 12, "TP": 64 },
  "feature_importance": { feature_name: importance_score, ... },
  "n_train": 608,
  "n_test": 152,
  "model_name": "Random Forest",
  "n_estimators": 200,
  "max_depth": 10
}
```

#### `feature_metadata.json`

```
Contains:
{
  "features": ["dem", "aspect", "slope", ...],      # Feature names
  "n_samples": 760,
  "n_per_class": 380,
  "reference_raster": "slope_modified.tif",
  "target_crs": "EPSG:32645",
  "raster_shape": [2281, 2134],
  "raster_bounds": [346263.46, 3054401.14, 408197.81, 3120601.83],  # minX, minY, maxX, maxY
  "raster_transform": [...],
  "soil_categories": { category_id: code, ... },
  "landuse_categories": { category_id: code, ... },
  "raster_files": {
    "dem": "Dem_sind.tif",
    "aspect": "aspect.tif",
    ...
  }
}
```

---

## Key Technologies & Dependencies

### Backend (Python)

| Package      | Version | Purpose                           |
| ------------ | ------- | --------------------------------- |
| FastAPI      | Latest  | Web framework for REST API        |
| Uvicorn      | Latest  | ASGI server (runs FastAPI)        |
| Rasterio     | Latest  | Read/write spatial rasters        |
| GeoPandas    | Latest  | Vector data manipulation          |
| Scikit-learn | Latest  | Machine learning (Random Forest)  |
| Joblib       | Latest  | Serialize/save models             |
| NumPy        | Latest  | Numerical computing               |
| Pandas       | Latest  | Tabular data manipulation         |
| Shapely      | Latest  | Geometric operations              |
| SciPy        | Latest  | Distance transforms, etc.         |
| PyProj       | Latest  | Coordinate system transformations |

### Frontend (JavaScript)

| Package  | Version | Purpose                 |
| -------- | ------- | ----------------------- |
| React    | 19.2.4  | UI framework            |
| Vite     | 8.0.2   | Build tool & dev server |
| Leaflet  | 1.9.4   | Interactive mapping     |
| Chart.js | 4.5.1   | Data visualization      |

### Geospatial Data Formats

- **.tif (GeoTIFF)**: Georeferenced rasters (DEM, slope, etc.)
- **.gpkg (GeoPackage)**: Vector database with geometries
- **.shp/.shx (Shapefile)**: Legacy vector format
- **.geojson**: Web-friendly vector format (JSON)

---

## Model Performance Summary

### What the Model Does

Given 10 geomorphological features at any location in Sindhupalchowk, the model predicts the probability of landslide occurrence.

### Accuracy Story

- **82.2% Accuracy**: Out of 152 test samples, model correctly classified 125
- **84.2% Recall**: Out of actual landslides, model catches 84 → few missed events
- **81% Precision**: Of predicted landslides, 81% true → conservative (avoids false alarms)
- **ROC-AUC 0.883**: Model is excellent at distinguishing landslide from non-landslide

### Limitations & Caveats

1. **Training Data Size**: Only 760 samples (limited by field survey resources)
2. **Class Imbalance**: Dataset is balanced artificially; real-world distribution unknown
3. **Temporal Aspect**: Model trained on historical data; future conditions may differ
4. **Spatial Variability**: Model assumes relationships are consistent across entire region
5. **Missing Features**: Climate data (rainfall) not included (difficult to obtain at pixel-level)
6. **Data Quality**: Vector data boundaries mapped manually → small inaccuracies

### How to Interpret a Prediction

When you click on a location and get probability = 0.73:

- **73% probability of landslide**: If this pixel were affected by the same environmental factors 100 times, ~73 times it would experience landslide
- **Not a guarantee**: 27% chance of no event (model uncertainty)
- **Use with context**: Combine with expert judgment, site inspection, rainfall patterns

---

## Workflow Diagram

### End-to-End Data Flow

```
Raw Geospatial Data
    ↓
[prepare_data.py]          → Pre-process & harmonize
    ↓
training_data.csv          → 760 balanced samples, 10 features
    ↓
[data.py]                  → Train Random Forest classifier
    ↓
model.joblib               → Serialized model
model_metrics.json         → Validation metrics
    ↓
[generate_susceptibility_map.py]  → Apply model to entire extent
    ↓
susceptibility_map.tif     → 0-1 probability map (4.8M pixels)
    ↓
[Backend: main.py]         → Load model & map, serve API
    ↓
API Endpoints
├─ /api/predict            → Point prediction
├─ /api/susceptibility-image  → Map visualization
├─ /api/landslides         → Historical locations
├─ /api/stats              → Summary statistics
└─ / (static files)        → React frontend
    ↓
[Frontend: React]          → Interactive map interface
    ↓
Web Browser                → User views predictions & explores data
    ↓
User clicks location       → POST /api/predict
    ↓
Backend extracts features, applies model, returns probability
    ↓
Frontend displays probability, risk category, extracted features
```

---

## Explanation for Professors

### Key Points to Emphasize

1. **Problem**: Landslides are a significant hazard in Nepal; traditional assessment is slow and expertise-dependent.

2. **Solution**: Automated ML model that learns patterns from:
   - Historical landslide locations (field survey data)
   - Geomorphological features (topography, slope, aspect, drainage)
   - Secondary features (soil, land use, proximity to rivers)

3. **Methodology**:
   - **Data Integration**: Harmonized 10+ geospatial datasets into unified framework
   - **Feature Engineering**: Extracted meaningful variables from raw rasters and vectors
   - **Model Selection**: Random Forest chosen for robustness and interpretability
   - **Validation**: 82% accuracy achieved with good precision-recall balance

4. **Deliverable**:
   - Probability map covering entire study area
   - Interactive web interface for exploration
   - Point-level predictions in real-time
   - Transparent decision-making (feature importance visible)

5. **Impact**:
   - Faster hazard assessment (replace weeks of fieldwork → seconds)
   - Reproducible, scientific approach
   - Scalable to other regions with similar data
   - Visual tool for communication with stakeholders

---

## Troubleshooting Guide

### Issue: Backend fails to start

**Solution**: Ensure all raster files exist and are readable. Check file paths in main.py.

### Issue: Frontend shows loading screen indefinitely

**Check**:

- Backend server running on port 8000?
- Network tab in browser dev tools for failed requests?
- Check browser console for errors

### Issue: Prediction returns NaN or error

**Causes**:

- Clicked outside study area boundary
- Nodata value in one of the feature rasters
  **Fix**: Click within Sindhupalchowk boundary

### Issue: Susceptibility map not showing

**Check**:

- susceptibility_map.tif file exists?
- Toggle visibility in LayersPanel?
- Check browser console

---

## Future Enhancements

1. **Add Climate Data**: Include rainfall, temperature (requires temporal downscaling)
2. **Temporal Analysis**: Track how susceptibility changes over time
3. **Risk Assessment**: Combine probability with exposure (population density) for risk
4. **Mobile App**: React Native version for field use
5. **Model Updates**: Retrain with new landslide data periodically
6. **Uncertainty Quantification**: Display confidence intervals alongside predictions
7. **API Authentication**: Secure endpoints if deploying publicly

---

## Contact & Documentation

- **Model Training Code**: data.py, prepare_data.py
- **Prediction Server**: backend/main.py
- **User Interface**: frontend/src/App.jsx
- **Validation Analysis**: validation.ipynb

---

_This documentation provides a complete overview for explaining your landslide susceptibility prediction system to professors, stakeholders, or collaborators. Every major component, data flow, and decision is explained with sufficient technical depth._
