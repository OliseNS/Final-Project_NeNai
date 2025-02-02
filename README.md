# EEG Clustering and Model Evaluation  for Data Mining Class

## Description  
This repository focuses on clustering EEG (electroencephalogram) features using the KMeans algorithm, alongside performance evaluation and visualization. The project includes feature generation, which transforms raw EEG data into meaningful representations for clustering. The primary aim is to assess clustering metrics (e.g., cohesion, separation, silhouette score) and visually compare results across datasets.  

### Key Features  
- **EEG Data Pre-processing**: Ensures clean and structured data for analysis.  
- **Feature Generation**: Extracts meaningful features from raw EEG signals to improve clustering performance.  
- **Clustering with KMeans**: Groups EEG features into distinct clusters.  
- **Performance Evaluation**: Metrics include:  
  - Silhouette Score  
  - Cohesion and Separation  
  - Recall, Specificity, and F1-Score  
- **Visualization**: Provides visual comparisons of clustering results across multiple datasets.  

## Requirements
Ensure you have the following installed:
```bash
pip install numpy pandas scikit-learn matplotlib
