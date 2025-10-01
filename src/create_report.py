from fpdf import FPDF
import os

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Report: Genetic Syndrome Classification', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_report():
    # Create PDF
    pdf = PDFReport()
    pdf.add_page()
    
    # Introduction
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. Introduction', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 10, 
        'This project aims to classify genetic syndromes based on image embeddings. '
        'The embeddings are 320-dimensional vectors from a pre-trained classification model. '
        'We used K-Nearest Neighbors (KNN) algorithms with different distance metrics '
        'to perform the syndrome classification.')
    pdf.ln()

    # Methodology
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. Methodology', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 10,
        '1. Data Loading and Preprocessing: The data was loaded from the file '
        'mini_gm_public_v0.1.p and transformed from a hierarchical structure '
        '(syndrome_id, subject_id, image_id) to embedding arrays and labels.\n\n'
        '2. Exploratory Analysis: Descriptive statistics of the dataset were calculated, '
        'including total number of samples, number of different syndromes, and '
        'sample distribution per syndrome.\n\n'
        '3. Visualization: The t-SNE technique was used to reduce embeddings from 320 dimensions '
        'to 2 dimensions, allowing visualization of the data in a two-dimensional space.\n\n'
        '4. Classification: The K-Nearest Neighbors (KNN) algorithm was implemented with two '
        'different distance metrics (Euclidean and Cosine). For each metric, we tested '
        'k values from 1 to 15.\n\n'
        '5. Cross-Validation: 10-fold cross-validation was used to evaluate '
        'model performance and determine the optimal k value.\n\n'
        '6. Evaluation Metrics: AUC (Area Under the ROC Curve), '
        'F1-Score, Accuracy, and Top-k Accuracy metrics were calculated to compare the performance '
        'of the different configurations.')
    pdf.ln()

    # Results
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '3. Results', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 10,
        '3.1. Exploratory Analysis:\n'
        '- Total number of samples: 1116\n'
        '- Number of different syndromes: 10\n'
        '- Average samples per syndrome: 111.60\n'
        '- Standard deviation of samples per syndrome: 51.27\n'
        '- Minimum samples per syndrome: 64\n'
        '- Maximum samples per syndrome: 210\n\n'
        '3.2. Classification:\n'
        'Euclidean Distance - Best k: 15, AUC: 0.9504, F1: 0.7547, Accuracy: 0.7634\n'
        'Cosine Distance - Best k: 15, AUC: 0.9630, F1: 0.7874, Accuracy: 0.7948\n\n'
        '3.3. Top-k Accuracy:\n'
        'Euclidean Distance (k=15):\n'
        '  Top-1 Accuracy: 0.7634, Top-3 Accuracy: 0.9247, Top-5 Accuracy: 0.9659\n'
        'Cosine Distance (k=15):\n'
        '  Top-1 Accuracy: 0.7948, Top-3 Accuracy: 0.9418, Top-5 Accuracy: 0.9749\n\n'
        '3.4. Conclusion:\n'
        'The Cosine distance metric performed better with an AUC of 0.9630, '
        'indicating that the angular similarity between embedding vectors is more '
        'appropriate for this task than Euclidean distance.')
    pdf.ln()

    # Analysis and Interpretation
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '4. Analysis and Interpretation', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 10,
        'The superior performance of the Cosine distance metric compared to Euclidean '
        'suggests that directional similarity between embeddings is more important '
        'for genetic syndrome classification than absolute distance in the space.\n\n'
        'The high AUC value (0.9630) for the Cosine metric indicates that the model '
        'is very effective at distinguishing between the different syndrome classes.\n\n'
        'Exploratory analysis revealed class imbalance, with some syndromes having '
        'more samples than others (64 to 210 samples). This imbalance may affect '
        'model performance, especially for minority classes.')
    pdf.ln()

    # Challenges and Solutions
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '5. Challenges and Solutions', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 10,
        'Challenges:\n'
        '- Imbalanced data: Some classes had significantly more samples than others\n'
        '- High dimensionality: 320-dimensional embeddings presented computational challenges\n'
        '- Multiclass complexity: Classification among 10 different classes required robust methods\n\n'
        'Solutions:\n'
        '- Use of appropriate metrics such as AUC, F1-Score, and Accuracy for multiclass evaluation\n'
        '- Cross-validation to obtain reliable performance estimates\n'
        '- t-SNE visualization for understanding data clustering')
    pdf.ln()

    # Recommendations
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '6. Recommendations for Improvements', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 10,
        '1. Increase dataset size for better representation of minority classes.\n'
        '2. Explore other distance metrics and classification algorithms.\n'
        '3. Apply balancing techniques such as SMOTE to handle class imbalance.\n'
        '4. Perform feature engineering to extract additional information from embeddings.\n'
        '5. Use ensemble techniques to improve model robustness.')
    pdf.ln()

    # Save PDF
    pdf.output('results/genetic_syndrome_classification_report.pdf')
    print("PDF report created successfully: results/genetic_syndrome_classification_report.pdf")

if __name__ == "__main__":
    create_report()