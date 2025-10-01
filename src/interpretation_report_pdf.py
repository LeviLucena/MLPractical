"""
Correção do relatório de interpretação para alinhar com as figuras descritas
Esta versão corrige o relatório para refletir as figuras teóricas especificadas
"""
from fpdf import FPDF
import os

class CorrectInterpretationReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Corrected Interpretation Report: Model Analysis Questions', 0, new_x='LMARGIN', new_y='NEXT', align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, new_x='RIGHT', new_y='TOP', align='C')

def create_correct_interpretation_report():
    # Create PDF
    pdf = CorrectInterpretationReport()
    pdf.add_page()
    
    # Introduction
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '1. Introduction', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10, 
        'This report addresses the interpretation questions related to model analysis with corrected figure associations. '
        'The theoretical figures described in the questions do not exactly match the actual generated figures from our '
        'genetic syndrome classification project. This report clarifies the mapping between theoretical concepts and '
        'actual results.')
    pdf.ln()

    # Question 1
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '2. Question 1: Figure 1 - Data distribution samples', 0, new_x='LMARGIN', new_y='NEXT')
    
    # Add Figure 1 right after the title
    try:
        pdf.ln(5)  # Add some space after the title
        pdf.image('results/tsne_visualization.png', x=10, y=None, w=170)
        pdf.ln(85)  # Add space after image
    except:
        pdf.set_font('helvetica', '', 10)
        pdf.cell(0, 10, 'Figure 1 image could not be loaded', 0, new_x='LMARGIN', new_y='NEXT')
        pdf.ln()
        pdf.set_font('helvetica', 'B', 12)
    
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10, 
        'Figure 1 presents a data distribution, the dots represent the sparse data for the axis X and Y, '
        'and the lines represent the fit of a hypothetical classification model.')
    pdf.ln()
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 10, '* Which distribution has the best balance between bias and variance?', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'Based on Figure 1, the distribution with the fitted line that captures the general trend of the data points '
        'without overfitting to the noise would have the best balance between bias and variance. '
        'This is typically a model with moderate complexity - not too simple (which would result in high bias and '
        'underfitting) and not too complex (which would result in high variance and overfitting). '
        'Ideally, it would be the model that follows the true underlying pattern in the data while maintaining '
        'generalization capability.')
    pdf.ln()
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 10, '* Describe your thoughts about your selection.', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'In the bias-variance tradeoff, a model with high bias tends to underfit the data (too simplistic), '
        'while a model with high variance tends to overfit the data (too complex and sensitive to noise). '
        'The optimal model minimizes both bias and variance simultaneously. The best balance is achieved '
        'when the model captures the underlying pattern without being overly sensitive to '
        'random fluctuations in the training data, which corresponds to a model that generalizes '
        'well to unseen data.')
    pdf.ln()

    # Question 2
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '3. Question 2: Figure 2 - Simple graph', 0, new_x='LMARGIN', new_y='NEXT')
    
    # Add Figure 2 right after the title
    try:
        pdf.ln(5)  # Add some space after the title
        pdf.image('results/roc_curves_comparison.png', x=10, y=None, w=170)
        pdf.ln(85)  # Add space after image
    except:
        pdf.set_font('helvetica', '', 10)
        pdf.cell(0, 10, 'Figure 2 image could not be loaded', 0, new_x='LMARGIN', new_y='NEXT')
        pdf.ln()
        pdf.set_font('helvetica', 'B', 12)
    
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'Figure 2 presents a simple graph with 2 curves and 1 line. In model selection and evaluation:')
    pdf.ln()
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 10, '* What is the purpose of this graph and its name?', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'Based on the distribution of Figure 2 - Simple graph, this is a bias-variance tradeoff graph '
        '(also known as a model complexity graph). The purpose is to illustrate how model performance changes '
        'with complexity. It typically shows training error decreasing with model complexity while '
        'validation/test error decreases initially but then increases after a certain point due to overfitting.')
    pdf.ln()
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 10, '* What kind of model result does the dashed line represent?', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'The dashed line in Figure 2 represents the optimal model complexity point - where the '
        'total error (bias + variance) is minimized. This is the point where the model '
        'achieves the best generalization performance, balancing between underfitting and overfitting.')
    pdf.ln()
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 10, '* Which curve represents a better fit, the red or the green? Why?', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'Based on Figure 2, the curve that represents a better fit would be the one that has lower error at the '
        'optimal complexity point (where the dashed line intersects). If one curve shows consistently lower '
        'validation error around the optimal point, it represents a better model. The better fit occurs at '
        'the point where the total error is minimized, typically where the validation error curve starts '
        'to rise while the training error continues to decrease.')
    pdf.ln()
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 10, '* Describe your thoughts about your selection.', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'Based on the distribution of Figure 2 - Simple graph, my selection depends on which curve demonstrates '
        'the classic bias-variance tradeoff more clearly. A well-designed model evaluation graph will show '
        'training error decreasing monotonically with complexity while validation error initially decreases '
        'but eventually increases due to overfitting. The sweet spot is just before overfitting begins, '
        'which represents the optimal balance between bias and variance.')
    pdf.ln()

    # Question 3
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '4. Question 3: Figure 3 - Model train and evaluation pipeline', 0, new_x='LMARGIN', new_y='NEXT')
    
    # Add Figure 3 right after the title
    try:
        pdf.ln(5)  # Add some space after the title
        pdf.image('results/metric_comparison.png', x=10, y=None, w=170)
        pdf.ln(85)  # Add space after image
    except:
        pdf.set_font('helvetica', '', 10)
        pdf.cell(0, 10, 'Figure 3 image could not be loaded', 0, new_x='LMARGIN', new_y='NEXT')
        pdf.ln()
        pdf.set_font('helvetica', 'B', 12)
    
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'Figure 3 presents a classification model training and the evaluation. This model classifies '
        '3 classes (A, B, C). Graph A represents the training accuracy over the epochs, Graph B represents '
        'the training loss over the epochs, and the table represents the evaluation of the model using some test samples, '
        'we used a confusion matrix to evaluate the classes trained.')
    pdf.ln()
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 10, '* Can we say that the model has a good performance in the test evaluation?', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'Based on the distribution of Figure 3 - Model train and evaluation pipeline, the model performance '
        'depends on the confusion matrix results. If the matrix shows high values on the diagonal '
        '(correct classifications) and low values off the diagonal (misclassifications), then yes, '
        'the model has good performance. Good performance would be indicated by high precision, '
        'recall, and F1 scores for all classes, showing that the model correctly identifies samples '
        'from classes A, B, and C.')
    pdf.ln()
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 10, '* What phenomenon happened during the test evaluation?', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'Based on the distribution of Figure 3 - Model train and evaluation pipeline, looking at the '
        'training accuracy and loss graphs, we would identify signs of overfitting or underfitting. '
        'Overfitting occurs when training accuracy continues to improve but validation/test performance '
        'starts to decline. Underfitting occurs when the model performs poorly on both training and validation data. '
        'If there\'s a significant gap between training and validation metrics, overfitting likely occurred.')
    pdf.ln()
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 10, '* Describe your thoughts about your selection.', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'Based on the distribution of Figure 3 - Model train and evaluation pipeline: If the training '
        'accuracy is high but validation accuracy is significantly lower, overfitting has occurred. '
        'This means the model learned the training data too well and lost its ability to generalize. '
        'Conversely, if both accuracies are low, underfitting occurred, indicating the model is too '
        'simple to capture the data patterns. The confusion matrix would provide insight into which '
        'classes the model struggles with the most, showing potential class imbalance issues or '
        'difficulties distinguishing between specific classes.')
    pdf.ln()

    # Note about figure correspondence
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '5. Mapping of Theoretical Figures to Actual Results', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'Note: The theoretical figures described in the questions do not exactly correspond to our actual project '
        'results. For this report, we are mapping the concepts as follows:')
    pdf.ln()
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 10, 'Theoretical Figure 1 (Data distribution) -> Actual Figure: t-SNE Visualization', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'The t-SNE visualization shows the distribution of genetic syndrome embeddings in a 2D space, '
        'which can conceptually represent "dots for sparse data" as described in the theoretical figure.')
    pdf.ln()
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 10, 'Theoretical Figure 2 (Model selection graph) -> Actual Figure: ROC Curves Comparison', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'While ROC curves are different from bias-variance tradeoff curves, both represent model selection and evaluation '
        '- comparing different models\' performance, which aligns with the conceptual purpose.')
    pdf.ln()
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 10, 'Theoretical Figure 3 (Training pipeline) -> Actual Figure: Metrics Comparison', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'The metrics comparison shows model performance across different k values, representing the '
        'evaluation phase of a model training pipeline, which aligns conceptually with the theoretical figure.')
    pdf.ln()

    # Include Remaining Results Images
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '6. Additional Results from Our Genetic Syndrome Classification', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'The following additional images show the actual results from our genetic syndrome classification project.')
    pdf.ln()

    # Add remaining images
    try:
        # Add distribution plot
        pdf.cell(0, 10, 'Sample Distribution by Syndrome:', 0, new_x='LMARGIN', new_y='NEXT')
        pdf.image('results/distribution_syndromes.png', x=10, y=None, w=170)
        pdf.ln(85)  # Add space after image
        
    except:
        pdf.set_font('helvetica', '', 10)
        pdf.multi_cell(0, 10, 'Note: Additional figure representations could not be loaded. The following files are required for the interpretation report:')
        pdf.cell(0, 10, '- results/distribution_syndromes.png', 0, new_x='LMARGIN', new_y='NEXT')
        pdf.ln()

    # Add our project results summary
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '7. Project Results Summary', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'Our genetic syndrome classification project using K-Nearest Neighbors achieved the following results:\\\\n\\\\n'
        '* Dataset: 1,116 samples across 10 different genetic syndromes\\\\n'
        '* Best Model: Cosine distance metric with k=15 achieved AUC of 0.9630\\\\n'
        '* Performance Comparison:\\\\n'
        '  - Euclidean Distance (k=15): AUC: 0.9504, F1: 0.7547, Accuracy: 0.7634\\\\n'
        '  - Cosine Distance (k=15): AUC: 0.9630, F1: 0.7874, Accuracy: 0.7948\\\\n'
        '* Top-k Accuracy Results:\\\\n'
        '  - Euclidean Distance: Top-1: 0.7634, Top-3: 0.9247, Top-5: 0.9659\\\\n'
        '  - Cosine Distance: Top-1: 0.7948, Top-3: 0.9418, Top-5: 0.9749\\\\n\\\\n'
        'The superior performance of cosine distance suggests that directional similarity between '
        'embedding vectors is more relevant for genetic syndrome classification than absolute '
        'Euclidean distance.')

    # Conclusion
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '8. Conclusion', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'This corrected interpretation report clarifies the mapping between theoretical concepts described in '
        'the analysis questions and our actual experimental results. While the exact figures differ, the '
        'underlying machine learning concepts of bias-variance tradeoff, model evaluation, and performance '
        'assessment remain applicable. Our genetic syndrome classification project demonstrates these '
        'principles in practice, achieving high performance with appropriate model selection techniques.')

    # Save PDF
    pdf.output('results/interpretation_analysis_report.pdf')
    print("Corrected PDF interpretation report created successfully: results/interpretation_analysis_report.pdf")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    create_correct_interpretation_report()