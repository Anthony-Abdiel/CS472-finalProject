# Determining Optimal Music Recommendation Models Using Clustering

## Overview
This project evaluates the effectiveness of various classical clustering algorithms on high-dimensional music data from the Free Music Archive (FMA).  
We investigate three dimensionality-reduction strategies — **PCA**, **UMAP**, and a **Hybrid PCA+UMAP** approach.  
Seven clustering algorithms were evaluated, silhouette scores were computed, and clustering behavior was visualized to analyze the structure of musical feature space.

---

## Key Objectives
- Evaluate clustering algorithms on high-dimensional music data  
- Compare PCA, UMAP, and hybrid dimensionality-reduction methods  
- Visualize cluster formations using PCA projections  
- Determine whether natural clusters emerge in musical feature space  
- Provide a reproducible experimental pipeline  

---

## Dataset
We used the **FMA metadata dataset**, specifically focusing on **`echonest.csv`**, which contains two major categories of features:

### 1. Audio Features
These include:
- Average timbre coefficients  
- Loudness  
- Tempo  
- Danceability  
- Other statistical summary descriptors  

### 2. Temporal Features
- Approximately **220 scalar values**  
- Describe how timbre evolves throughout the song  
- Flattened into a single high-dimensional feature vector  

---

## Usage Instructions

### 1. Set up the Python environment
Requires **Python 3.11.0**.

```bash
python3 -m venv music_env
source music_env/bin/activate   # Mac/Linux
.\music_env\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 2. Run the project

```bash
python main.py
```

### 3. Outputs
Running the script generates:
- Silhouette score metrics printed to the terminal  
- Visualization graphs including:
  - PCA scatter plots  
  - Silhouette plots  
  - HDBSCAN condensed cluster trees  
  - Elbow and AIC/BIC curves  

---

## Limitations
- Music data is noisy, highly multidimensional, and lies on a continuous manifold, making clustering difficult  
- Generated clusters may not align cleanly with human-labeled genres  
- Silhouette score alone cannot fully evaluate cluster “correctness”  
- UMAP projections can vary depending on selected hyperparameters  

---

## Authors
**Anthony Narvaez**  
**Benjamin Huntoon**

---

## License
MIT
