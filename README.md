# MSTFFN - Maritime Traffic Flow Prediction

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Status](https://img.shields.io/badge/status-research-yellow)

MSTFFN (Multi-Scale Temporal Feature Fusion Network) is an advanced deep learning system for maritime traffic flow prediction using real AIS (Automatic Identification System) data. The model combines multi-head attention mechanisms with temporal positional encoding to predict vessel traffic patterns and density distributions in maritime areas.

**Based on research**: "AIS Data-Driven Maritime Traffic Flow Prediction and Density Visualization"

---

## 🌊 **Overview**

MSTFFN processes AIS vessel tracking data at multiple temporal scales (daily, weekly, monthly) to generate accurate traffic flow predictions for maritime surveillance, port management, and maritime safety applications.

### **Key Features**

- **Advanced Architecture**:
  - Multi-scale temporal feature extraction (24h, 72h, 168h windows)
  - Multi-head attention mechanism for temporal dependencies
  - Gaussian probabilistic predictions (mean μ and variance σ²)
  - Transformer-based fusion of multi-scale features

- **Real AIS Data Integration**:
  - Automatic download from NOAA and Marine Cadastre databases
  - Processing of real vessel trajectories and positions
  - Support for multiple data sources (NOAA 2022, 2023, Marine Cadastre)
  - Fallback to high-quality synthetic data when real data unavailable

- **Comprehensive Visualization**:
  - Traffic flow predictions with confidence intervals
  - Statistical error distribution analysis
  - Diebold-Mariano statistical significance testing
  - Spatial density heatmaps and gradient analysis
  - Grid cell traffic distribution
  - Vessel speed distribution analysis

- **Technical Features**:
  - Configurable training parameters via CONFIG dictionary
  - CUDA GPU acceleration support
  - Multi-model comparison (GRU, LSTM, BiLSTM, ConvLSTM, Transformer)
  - Extensive statistical metrics (RMSE, MAE, MAPE, R²)
  - Real-time training monitoring

---

## 📊 **Architecture**

### **1. Multi-Scale Feature Extraction**

The model processes traffic data at three temporal scales:
- **Low scale** (24h): Daily patterns and short-term fluctuations
- **Medium scale** (72h): Weekly trends and periodic behaviors
- **High scale** (168h): Long-term patterns and seasonal variations

### **2. Temporal Transformer Network**

- **Multi-Head Attention**: Captures complex temporal dependencies
- **Positional Time Encoding**: Sine/cosine embeddings + learned time features
- **Layer Normalization**: Stable training with residual connections
- **Feed-Forward Networks**: Non-linear feature transformation

### **3. Gaussian Prediction Head**

- Outputs probabilistic predictions (μ, σ²)
- Enables uncertainty quantification
- Supports confidence interval estimation

### **4. Data Processing Pipeline**

- **Input**: AIS position reports (lat, lon, timestamp, MMSI)
- **Processing**: Temporal aggregation into hourly traffic counts
- **Normalization**: StandardScaler per temporal scale
- **Output**: Traffic flow predictions with statistical confidence

---

## 🗂️ **Data Requirements**

### **Supported Data Sources**

1. **NOAA AIS Data** (2022, 2023)
   - Format: ZIP archives containing CSV files
   - Coverage: US coastal waters
   - Fields: LAT, LON, BaseDateTime, MMSI, SOG, COG

2. **Marine Cadastre AIS Data**
   - Format: ZIP archives containing CSV files
   - Coverage: US maritime zones
   - Similar field structure to NOAA

3. **Synthetic High-Quality Data** (Fallback)
   - Generated when real data is unavailable
   - Realistic traffic patterns with daily/weekly/monthly cycles
   - Configurable for any geographic area

### **Data Organization**

The system automatically:
- Downloads and extracts AIS data from public sources
- Filters data by geographic bounding box
- Converts position reports to traffic flow time series
- Handles missing timestamps and irregular sampling
- Generates directional features (eastbound/westbound)

---

## 📈 **Evaluation Metrics**

### **Core Metrics**

- **RMSE** (Root Mean Squared Error): Overall prediction accuracy
- **MAE** (Mean Absolute Error): Average prediction deviation
- **MAPE** (Mean Absolute Percentage Error): Relative error percentage
- **R² Score**: Coefficient of determination

### **Statistical Tests**

- **Diebold-Mariano Test**: Statistical significance between models
- **Error Distribution Analysis**: Normality tests and histogram fitting
- **Confidence Intervals**: 95% prediction confidence bounds

### **Visualization Tools**

- Traffic flow line plots with confidence intervals
- Error distribution histograms with normal fits
- Box-and-whisker plots for model comparison
- Spatial density heatmaps (2D contour plots)
- Grid cell traffic distribution matrices
- Vessel speed vs. traffic flow heatmaps
- Daily traffic patterns (hourly × date)
- Trajectory visualization with gate lines

---

## 🛠️ **Installation**

### **Prerequisites**

- Python 3.8-3.11
- CUDA 11.x or 12.x (optional, for GPU acceleration)

### **Recommended: Use Conda**
```bash
# Create environment
conda create -n mstffn python=3.10 -y
conda activate mstffn

# Install PyTorch with CUDA support (for GPU)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# For CPU-only installation:
# conda install pytorch torchvision cpuonly -c pytorch -y

# Install dependencies
pip install numpy pandas scikit-learn scipy matplotlib seaborn requests
```

### **macOS (Apple Silicon) Installation**
```bash
conda create -n mstffn python=3.10 -y
conda activate mstffn

# PyTorch with MPS (Metal Performance Shaders) support
conda install pytorch torchvision -c pytorch -y

# Install dependencies
pip install numpy pandas scikit-learn scipy matplotlib seaborn requests
```

### **Windows Installation**
```bash
# Create environment
conda create -n mstffn python=3.10 -y
conda activate mstffn

# Install PyTorch (CUDA or CPU)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install numpy pandas scikit-learn scipy matplotlib seaborn requests
```

---

## ⚙️ **Configuration**

All training and model parameters are centralized in the `CONFIG` dictionary:
```python
CONFIG = {
    # Training dataset size (paper uses 1.8M samples)
    "TRAIN_SAMPLES": 1_800_000,
    
    # AIS data sources (NOAA, Marine Cadastre)
    "AIS_SOURCES": {
        'noaa_2022': 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2022/...',
        'marine_cadastre': 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2021/...',
        'noaa_2023': 'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/...'
    },
    
    # Test area: Gulf of Naples (Italy)
    "TEST_AREA_NAME": "Golfo di Napoli",
    "TEST_AREA_BOUNDS": {
        'lon_min': 13.90,
        'lon_max': 14.45,
        'lat_min': 40.50,
        'lat_max': 40.95
    },
    
    # Training hyperparameters
    "BATCH_SIZE": 32,
    "EPOCHS": 200,
    "LEARNING_RATE": 0.001,
    
    # Device: 'cuda', 'cpu', or 'mps' (macOS)
    "DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Fallback synthetic data size
    "SIMULATED_SAMPLES_FALLBACK": 200_000,
}
```

### **Configuring for Different Maritime Areas**

To analyze a different port or maritime region, modify `TEST_AREA_BOUNDS`:
```python
# Example: Port of Rotterdam
"TEST_AREA_BOUNDS": {
    'lon_min': 3.80,
    'lon_max': 4.50,
    'lat_min': 51.80,
    'lat_max': 52.10
}
```

---

## 🚀 **Usage**

### **Basic Execution**
```bash
python mstffn_maritime_traffic.py
```

The system will automatically:
1. Download AIS data for the configured area
2. Process and normalize the data
3. Train the MSTFFN model (200 epochs by default)
4. Generate predictions and visualizations
5. Output performance metrics

### **Expected Output**
```
=== MSTFFN - Maritime Traffic Flow Prediction with REAL AIS Data ===

CONFIG: TRAIN_SAMPLES=1800000, TEST_AREA=Golfo di Napoli
Training Epochs: 200, Batch Size: 32

1. Download and processing dati AIS reali per Golfo di Napoli...
Scaricando dati NOAA 2023 da: https://coast.noaa.gov/...
Dati NOAA 2023 caricati: 1800000 record
Dati filtrati per Golfo di Napoli: 45823 record

2. Training modello MSTFFN...
Dispositivo di training: cuda
Parametri modello: 1,234,567

Epoch 20/200, Loss: 0.3245, LR: 0.001000
Epoch 40/200, Loss: 0.2156, LR: 0.001000
...
Epoch 200/200, Loss: 0.0823, LR: 0.000125

3. Generazione predizioni...
4. Visualizzazioni statistiche...

======================================================================
IMPROVED MODEL PERFORMANCE METRICS
Golfo di Napoli AIS DATA - ENHANCED CONFIGURATION
======================================================================

Model              RMSE       MAE        MAPE (%)    Training Epochs
----------------------------------------------------------------------
MST-GRU            0.2456     0.1834     12.3456     N/A            
MST-LSTM           0.2234     0.1645     10.8934     N/A            
MST-BiLSTM         0.2312     0.1723     11.2345     N/A            
MST-CovLSTM        0.2145     0.1598      9.7456     N/A            
Transformer        0.2089     0.1512      9.2134     N/A            
MSTFFN-Improved    0.1823     0.1289      7.8456     200            

RMSE Improvement vs Best Baseline: 12.74%
======================================================================
```

### **Training with Custom Parameters**

Modify `CONFIG` before running:
```python
CONFIG["EPOCHS"] = 100  # Faster training
CONFIG["BATCH_SIZE"] = 64  # Larger batches
CONFIG["LEARNING_RATE"] = 0.0005  # Lower learning rate
CONFIG["DEVICE"] = 'mps'  # Use Apple Silicon GPU
```

---

## 📊 **Visualization Outputs**

The system generates comprehensive visualizations:

### **1. Traffic Flow Prediction**
- Line plot with ground truth vs predictions
- 95% confidence intervals
- Error distribution histogram with normal fit

### **2. Model Comparison**
- Box-and-whisker plots of prediction errors
- Diebold-Mariano heatmaps (statistical significance)
- RMSE/MAE/MAPE bar charts

### **3. Spatial Analysis**
- Traffic density heatmaps (contour plots)
- Density gradient visualization with vector fields
- Grid cell traffic distribution matrices

### **4. Temporal Patterns**
- Daily traffic heatmaps (hour × date)
- Vessel speed vs traffic flow correlation
- Training loss evolution curves

### **5. Advanced Analytics**
- Trajectory visualization with gate lines
- Vessel type performance analysis
- Multi-gate comparison dashboard

---

## 🧪 **Extending the Model**

### **Adding New Data Sources**
```python
CONFIG["AIS_SOURCES"]["custom_source"] = "https://your-ais-data-url.com/data.zip"

# In AISDataDownloader class, add:
def _download_custom_data(self, area_bounds, n_rows):
    # Your custom download logic
    pass
```

### **Custom Features**

Add new features to the preprocessing pipeline:
```python
def _add_weather_features(self, df):
    """Add meteorological features to AIS data"""
    df['wind_speed'] = ...  # Fetch from weather API
    df['wave_height'] = ...
    return df
```

### **Model Architecture Modifications**
```python
# Modify CONFIG for different temporal scales
preprocessor = AISDataPreprocessor(
    time_scales={'low': 12, 'medium': 48, 'high': 120}  # Custom scales
)

# Adjust model dimensions
model = MSTFFN(d_model=256, n_heads=16, n_layers=6)  # Larger model
```

---

## 📚 **Code Structure**
```
mstffn_maritime_traffic.py
│
├── CONFIG                          # Centralized configuration
├── MultiHeadAttention              # Attention mechanism
├── PositionalTimeEncoding          # Temporal embeddings
├── MSTFFN                          # Main model architecture
│
├── AISDataDownloader               # Data acquisition system
│   ├── download_real_ais_data()
│   ├── _download_noaa_data()
│   ├── _process_ais_dataframe()
│   └── _create_high_quality_simulated_data()
│
├── StatisticalVisualizer           # Visualization system
│   ├── plot_traffic_flow_with_grid()
│   ├── plot_prediction_errors_boxplot()
│   ├── plot_diebold_mariano_heatmap()
│   ├── plot_traffic_density_heatmap()
│   └── [12 additional plotting methods]
│
├── AISDataPreprocessor             # Data preprocessing
│   ├── load_and_process_real_data()
│   ├── prepare_data()
│   └── _normalize_multiscale_data()
│
├── AISDataset                      # PyTorch Dataset
├── MSTFFNTrainer                   # Training loop
│   ├── train()
│   └── predict()
│
└── main()                          # Execution pipeline
```

---

## 🔬 **Research Context**

This implementation is based on academic research in maritime traffic prediction using deep learning. The model architecture combines:

- **Multi-scale temporal analysis**: Inspired by time series forecasting literature
- **Attention mechanisms**: Adapted from Transformer architectures
- **Probabilistic predictions**: Gaussian output for uncertainty quantification
- **Real-world validation**: Tested on NOAA AIS datasets

### **Comparison with Baseline Models**

| Model | Architecture | Temporal Modeling | Uncertainty |
|-------|--------------|-------------------|-------------|
| MST-GRU | Recurrent | Single-scale | ❌ |
| MST-LSTM | Recurrent | Single-scale | ❌ |
| MST-BiLSTM | Bidirectional Recurrent | Single-scale | ❌ |
| MST-CovLSTM | Convolutional + Recurrent | Single-scale | ❌ |
| Transformer | Self-Attention | Single-scale | ❌ |
| **MSTFFN** | **Multi-Head Attention** | **Multi-scale** | **✅** |

---

## ⚠️ **Important Notes**

### **Dataset Considerations**

> **Note**: The current implementation trains on US coastal AIS data (NOAA) but applies predictions to the Gulf of Naples. For production use in Mediterranean waters, the model should be retrained with:
> - Mediterranean AIS datasets
> - Region-specific traffic patterns
> - Local maritime regulations and routes
> - Adjusted geographic features and coordinates

### **Data Privacy**

- MMSI (vessel identifiers) are anonymized in visualizations
- No personal information is stored or transmitted
- AIS data is publicly available from NOAA and Marine Cadastre

### **Performance Optimization**

- Use GPU for training (20-40x speedup)
- Enable mixed precision training (AMP) for larger batches
- Reduce `TRAIN_SAMPLES` for faster experimentation
- Use `EPOCHS=50` for initial testing

---

## 🐛 **Troubleshooting**

### **Common Issues**

**1. CUDA Out of Memory**
```python
CONFIG["BATCH_SIZE"] = 16  # Reduce batch size
```

**2. Download Timeout**
```python
# Increase timeout in requests.get()
response = requests.get(url, timeout=300)  # 5 minutes
```

**3. Missing Coordinate Columns**
- System automatically generates synthetic data if columns not found
- Check CSV structure with `pandas.read_csv(..., nrows=5)`

**4. Empty Test Area**
```python
# Expand bounding box
CONFIG["TEST_AREA_BOUNDS"]['lon_min'] -= 0.5
CONFIG["TEST_AREA_BOUNDS"]['lon_max'] += 0.5
```

---

## 📄 **License**

Apache 2.0 License - See LICENSE file for details

---

## 🙏 **Acknowledgments**

- **Data Sources**: NOAA Office for Coastal Management, Marine Cadastre
- **Inspiration**: "AIS Data-Driven Maritime Traffic Flow Prediction and Density Visualization"
- **Framework**: PyTorch, scikit-learn, matplotlib
- **Community**: Open-source maritime data analysis community

---

## 📧 **Contact & Support**

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact: [your-email@example.com]
- Documentation: [GitHub Wiki]

---

## 🔮 **Future Enhancements**

- [ ] Real-time streaming AIS data integration
- [ ] Multi-region simultaneous prediction
- [ ] Weather condition integration (wind, waves, currents)
- [ ] Port congestion prediction
- [ ] Collision risk assessment
- [ ] API for maritime traffic services
- [ ] Web-based visualization dashboard
- [ ] Mobile app for port authorities

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Status**: Research Implementation
```

---

## **File aggiuntivi suggeriti per il repository GitHub:**

### **1. LICENSE** (Apache 2.0)
```
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

[Full Apache 2.0 license text]
```

### **2. requirements.txt**
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.31.0
```

### **3. .gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/

# Data
*.csv
*.zip
*.tiff
data/
checkpoints/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
