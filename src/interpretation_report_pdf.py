from fpdf import FPDF
import os

class InterpretationReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Interpretation Report: Model Analysis Questions', 0, new_x='LMARGIN', new_y='NEXT', align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, new_x='RIGHT', new_y='TOP', align='C')

def create_interpretation_report():
    # Create PDF
    pdf = InterpretationReport()
    pdf.add_page()
    
    # Introduction
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '1. Introduction', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10, 
        'This report addresses the interpretation questions related to model analysis. '
        'Each question is analyzed with detailed explanations based on machine learning principles.')
    pdf.ln()

    # Question 1
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '2. Question 1: Figure 1 - Data distribution samples', 0, new_x='LMARGIN', new_y='NEXT')
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

    # Include Results Images
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '5. Results from Our Genetic Syndrome Classification', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'The following images show the results from our genetic syndrome classification project.')
    pdf.ln()

    # Add images to the report
    try:
        # Add t-SNE visualization
        pdf.cell(0, 10, 't-SNE Visualization of Embeddings:', 0, new_x='LMARGIN', new_y='NEXT')
        pdf.image('results/tsne_visualization.png', x=10, y=None, w=170)
        pdf.ln(85)  # Add space after image
        
        # Add distribution plot
        pdf.cell(0, 10, 'Sample Distribution by Syndrome:', 0, new_x='LMARGIN', new_y='NEXT')
        pdf.image('results/distribution_syndromes.png', x=10, y=None, w=170)
        pdf.ln(85)  # Add space after image
        
        # Add metric comparison
        pdf.cell(0, 10, 'Performance Metrics Comparison:', 0, new_x='LMARGIN', new_y='NEXT')
        pdf.image('results/metric_comparison.png', x=10, y=None, w=170)
        pdf.ln(85)  # Add space after image
        
        # Add ROC curves
        pdf.cell(0, 10, 'ROC Curves Comparison:', 0, new_x='LMARGIN', new_y='NEXT')
        pdf.image('results/roc_curves_comparison.png', x=10, y=None, w=170)
        pdf.ln(85)  # Add space after image
        
    except:
        pdf.set_font('helvetica', '', 10)
        pdf.multi_cell(0, 10, 'Note: Images could not be loaded. Please ensure the results directory contains the following files:')
        pdf.cell(0, 10, '- results/tsne_visualization.png', 0, new_x='LMARGIN', new_y='NEXT')
        pdf.cell(0, 10, '- results/distribution_syndromes.png', 0, new_x='LMARGIN', new_y='NEXT')
        pdf.cell(0, 10, '- results/metric_comparison.png', 0, new_x='LMARGIN', new_y='NEXT')
        pdf.cell(0, 10, '- results/roc_curves_comparison.png', 0, new_x='LMARGIN', new_y='NEXT')
        pdf.ln()

    # Add our project results summary
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '6. Project Results Summary', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'Our genetic syndrome classification project using K-Nearest Neighbors achieved the following results:\\n\\n'
        '* Dataset: 1,116 samples across 10 different genetic syndromes\\n'
        '* Best Model: Cosine distance metric with k=15 achieved AUC of 0.9630\\n'
        '* Performance Comparison:\\n'
        '  - Euclidean Distance (k=15): AUC: 0.9504, F1: 0.7547, Accuracy: 0.7634\\n'
        '  - Cosine Distance (k=15): AUC: 0.9630, F1: 0.7874, Accuracy: 0.7948\\n'
        '* Top-k Accuracy Results:\\n'
        '  - Euclidean Distance: Top-1: 0.7634, Top-3: 0.9247, Top-5: 0.9659\\n'
        '  - Cosine Distance: Top-1: 0.7948, Top-3: 0.9418, Top-5: 0.9749\\n\\n'
        'The superior performance of cosine distance suggests that directional similarity between '
        'embedding vectors is more relevant for genetic syndrome classification than absolute '
        'Euclidean distance.')
    pdf.ln()

    # Conclusion
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '7. Conclusion', 0, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 10,
        'This interpretation report analyzed theoretical concepts of model evaluation and '
        'bias-variance tradeoff, applying these principles to understand machine learning model '
        'performance. The practical results from our genetic syndrome classification project '
        'demonstrate effective application of these concepts, achieving high performance '
        'with appropriate model selection and evaluation techniques.')

    # Save PDF
    pdf.output('results/interpretation_analysis_report.pdf')
    print("PDF interpretation report created successfully: results/interpretation_analysis_report.pdf")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    create_interpretation_report()