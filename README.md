<p align="center">

  <!-- Linguagem principal -->
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  </a>

  <!-- Data Science e ML -->
  <a href="https://numpy.org/">
    <img src="https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" />
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/-Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn" />
  </a>
  <a href="https://pandas.pydata.org/">
    <img src="https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas" />
  </a>
  <a href="https://www.scipy.org/">
    <img src="https://img.shields.io/badge/-SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white" alt="SciPy" />
  </a>

  <!-- Visualização -->
  <a href="https://matplotlib.org/">
    <img src="https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square&logo=python&logoColor=white" alt="Matplotlib" />
  </a>
  <a href="https://seaborn.pydata.org/">
    <img src="https://img.shields.io/badge/-Seaborn-3776AB?style=flat-square&logo=python&logoColor=white" alt="Seaborn" />
  </a>

  <!-- Geração de PDF -->
  <a href="https://pypi.org/project/fpdf2/">
    <img src="https://img.shields.io/badge/-FPDF2-005F6A?style=flat-square&logo=python&logoColor=white" alt="FPDF2" />
  </a>

  <!-- Status do projeto -->
  <img src="https://img.shields.io/badge/status-completo-success?style=flat-square" alt="Status" />

  <!-- Licença -->
  <img src="https://img.shields.io/badge/license-License%20Not%20Specified-blue?style=flat-square" alt="License" />

</p>

# Genetic Syndrome Classification Project

This project implements a machine learning solution to classify genetic syndromes using image embeddings. The embeddings are 320-dimensional vectors from a pre-trained classification model.
<img width="1024" height="576" alt="Gemini_Generated_Image_ly34q2ly34q2ly341" src="https://github.com/user-attachments/assets/9ebb53dd-f916-46c1-92c9-64cdf1976ab2" />

## Project Structure

- `src/`: Source code for the project
  - `ml_project.py`: Main script for data processing, analysis and classification
  - `create_report.py`: Script to generate the final PDF report
  - `interpretation_report_pdf.py`: Script to generate interpretation
- `results/`: Output files including plots and reports
- `mini_gm_public_v0.1.p`: Input dataset file (provided)

## Requirements

- Python 3.13
- Required packages: numpy, scikit-learn, matplotlib, seaborn, pandas, fpdf2

## Installation & Setup

1. Install Python 3.13
2. Install required packages:
   ```
   pip install -r requirements.txt
   
   ```
3. Place the `mini_gm_public_v0.1.p` file in the project root directory

## How to Run

1. Execute the main analysis script:
   ```
   python src/ml_project.py
   ```
2. Generate the PDF report:
   ```
   python src/create_report.py
   ```
3. Execute Script to generate interpretation
   ```
   python src/interpretation_report_pdf.py
   ```

## Results

The project output includes:

- **Distribution Plot**: Shows sample distribution across different syndromes
- **t-SNE Visualization**: 2D visualization of embeddings grouped by syndrome
- **Metric Comparison**: Performance comparison of different k values for both distance metrics
- **ROC Curves**: Average ROC curves comparing Euclidean and Cosine distance metrics
- **PDF Report**: Comprehensive report with methodology, results and analysis

### Key Findings

- **Dataset**: 1,116 samples across 10 different genetic syndromes
- **Best Model**: Cosine distance metric with k=15 achieved AUC of 0.9630
- **Performance**: Cosine distance outperformed Euclidean distance in all metrics

### Classification Results

- **Euclidean Distance (k=15)**: AUC: 0.9504, F1: 0.7547, Accuracy: 0.7634
- **Cosine Distance (k=15)**: AUC: 0.9630, F1: 0.7874, Accuracy: 0.7948

## Methodology

1. **Data Loading**: Load hierarchical data from pickle file
2. **Preprocessing**: Flatten hierarchical structure to embeddings and labels
3. **Exploratory Analysis**: Calculate statistics and distribution of syndromes
4. **Visualization**: Use t-SNE to visualize embeddings in 2D
5. **Classification**: Implement KNN with Euclidean and Cosine distance metrics
6. **Cross-Validation**: 10-fold cross-validation to determine optimal k value
7. **Evaluation**: Calculate AUC, F1-Score, and Accuracy metrics
8. **Comparison**: Compare performance between distance metrics

## Files Generated

- `distribution_syndromes.png`: Sample distribution by syndrome
- `tsne_visualization.png`: t-SNE visualization of embeddings
- `metric_comparison.png`: Performance comparison of metrics
- `roc_curves_comparison.png`: ROC curves for both metrics
- `genetic_syndrome_classification_report.pdf`: Comprehensive analysis report
- `interpretation_analysis_report.pdf`: Interpretation analysis report

## Analysis

The superior performance of cosine distance suggests that directional similarity between embedding vectors is more relevant for genetic syndrome classification than absolute Euclidean distance. The high AUC score of 0.9630 demonstrates the effectiveness of the approach.
