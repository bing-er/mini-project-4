# Fashion-MNIST Classifier for StyleSort

## Project Overview
StyleSort, an online fashion retailer, faces a 32% return rate due to product miscategorization. This project develops a deep learning classifier to automatically categorize fashion items into 10 categories, reducing costly misclassifications and improving customer satisfaction.

## Business Problem
- **Current Issue**: 40% of returns due to "item wasn't what I expected"
- **Cost**: Misclassifications lead to customer confusion, returns, and lost revenue
- **Goal**: Achieve >85% accuracy with confidence-based human-in-the-loop review

## Results Summary

### Model Performance
| Experiment | Architecture | Test Accuracy | Key Features |
|------------|--------------|---------------|--------------|
| Baseline | 784→128→10 | 87.88%        | Simple 2-layer network |
| Deep Network | 784→256→128→64→10 | **88.96%**    | Dropout + BatchNorm |
| Lower LR | 784→128→10 | 84.64%        | Learning rate 0.001 |

**Best Model**: Deep Network (88.96% accuracy) ✓ Exceeds 85% requirement by 3.96 percentage points

### Business Impact
- **Cost-Weighted Accuracy**: 92.02%
  - Model errors cost only 8.0% of maximum possible misclassification cost
- **Recommended Confidence Threshold**: 0.80
  - 96.00% accuracy on accepted predictions
  - Auto-processes 76.69% of items (7,669/day)
  - 23.31% flagged for human review (2,331/day)

### Alternative Thresholds
| Threshold | Accuracy on Accepted | Coverage | Items Automated (10k/day) | Items for Review |
|-----------|---------------------|----------|--------------------------|------------------|
| 0.70 (Aggressive) | 94.38% | 83.10% | 8,310 | 1,690 |
| **0.80 (Recommended)** | **96.00%** | **76.69%** | **7,669** | **2,331** |
| 0.90 (Conservative) | 97.78% | 67.15% | 6,715 | 3,285 |
| 0.95 (Very Conservative) | 98.63% | 59.81% | 5,981 | 4,019 |

### Key Insights
1. **Most Confused Pairs (Deep Network)**:
   - Pullover → Coat (118 errors) - High business cost
   - T-shirt → Shirt (138 errors) - High business cost  
   - Shirt → Pullover (83 errors) - Medium-high business cost

2. **Best Performing Categories**: 
   - Trouser (99% precision, 97% recall)
   - Bag (98% precision, 98% recall)
   - Sandal (99% precision, 94% recall)

3. **Challenging Categories**: 
   - Shirt (71% precision, 71% recall) - needs improved photography guidelines
   - Pullover (82% precision, 80% recall) - often confused with Coat

## Business Recommendations

### Immediate Actions

1. **Deploy with 0.80 confidence threshold** (Recommended)
   - **Automation**: Automatically categorize 76.69% of items (7,669/day)
   - **Accuracy**: 96.00% accuracy on auto-categorized items
   - **Quality Control**: Route 23.31% (2,331/day) to human reviewers
   - **Expected Errors**: ~307 misclassifications per day on automated items
   - **Business Value**: Reduces manual categorization workload by 77%

2. **Improve Product Photography for Confused Categories**
   - **Pullover vs. Coat**: Show fabric weight, lining, intended use, and warmth rating
   - **Shirt vs. T-shirt**: Emphasize collar type, buttons, formality level
   - **Coat vs. Pullover**: Highlight material thickness and season appropriateness
   - Provide standardized photography guidelines to vendors

3. **Enhanced Descriptions for High-Cost Pairs**
   - Add "Style: Casual/Formal" tags for tops
   - Include "Warmth Rating: 1-5" for outerwear
   - Specify "Season: Summer/Winter/All-Season"
   - Add "Collar Type: Button-down/Crew neck/V-neck" for shirts

### Deployment Options

**Option A: Balanced (Recommended - 0.80 threshold)**
- Best balance between automation (77%) and quality (96% accuracy)
- Manageable human review workload (2,331 items/day)
- Suitable for production launch

**Option B: Aggressive (0.70 threshold)**
- Maximizes automation (83% of items)
- Slightly lower accuracy (94.38%)
- Use if manual review resources are very limited

**Option C: Conservative (0.90 threshold)**
- Highest quality (97.78% accuracy)
- Lower automation (67% of items)
- Use during initial rollout or for high-value product categories

### Future Improvements
- **Enhanced Training Data**: Collect more samples for Shirt category (currently lowest performer)
- **Data Augmentation**: Implement rotation, brightness, and contrast adjustments
- **Architecture Upgrade**: Explore CNN architectures (ResNet, EfficientNet) for better feature extraction
- **Higher Resolution**: Train on higher-resolution color images for better detail recognition
- **A/B Testing**: Test confidence thresholds in production and adjust based on actual return rates
- **Ensemble Methods**: Combine multiple models for improved predictions on ambiguous items
- **Active Learning**: Prioritize human review on items most likely to improve model performance

## Setup Instructions

### Requirements
```bash
pip install -r requirements.txt
```

**Required packages:**
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
seaborn>=0.12.0
jupyter>=1.0.0
tqdm>=4.65.0
```

### Running the Project

**Option 1: Jupyter Notebook** (Recommended)
```bash
# Start Jupyter
jupyter notebook

# Open notebooks/fashion_classifier.ipynb
```

**Option 2: Python Scripts**
```bash
# Train a model
python src/train.py

# Or import and use
from src.model import FashionClassifier, DeepFashionClassifier
from src.train import train_model
```

### Project Structure
```
mini-project-4/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── fashion_classifier.ipynb    # Main analysis notebook
├── src/
│   ├── model.py                     # Neural network architectures
│   ├── train.py                     # Training loops and evaluation
│   └── utils.py                     # Data loading and helpers
└── results/
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── confusion_matrix_deep_network.png
    ├── cost_matrix.png
    ├── confidence_threshold_analysis.png
    ├── misclassified_examples.png
    ├── experiment_comparison.png
    ├── baseline_model.pth
    ├── deep_model.pth
    └── experiment_results.csv
```

## Technical Details

### Model Architecture (Best Performing - Deep Network)
```
Input: 28×28 grayscale images (784 pixels)
│
├─ Dense Layer 1: 256 neurons
├─ Batch Normalization
├─ ReLU Activation
├─ Dropout (0.3)
│
├─ Dense Layer 2: 128 neurons
├─ Batch Normalization
├─ ReLU Activation
├─ Dropout (0.3)
│
├─ Dense Layer 3: 64 neurons
├─ Batch Normalization
├─ ReLU Activation
├─ Dropout (0.3)
│
└─ Output Layer: 10 classes (softmax)

Total Parameters: 243,658
Optimizer: Adam (lr=0.001)
Loss: CrossEntropyLoss
Training Time: ~5 minutes (CPU)
```

### Training Configuration
- **Dataset**: Fashion-MNIST (70,000 total images)
- **Train/Val/Test Split**: 54,000 / 6,000 / 10,000
- **Batch Size**: 64
- **Epochs**: 15 (Deep Network), 10 (Baseline)
- **Hardware**: CPU (Apple Silicon / Intel)
- **Validation Strategy**: 10% holdout from training set
- **Random Seed**: 42 (for reproducibility)

### Model Performance Details

**Deep Network (88.96% Test Accuracy):**
```
Per-Class Performance:
                  Precision  Recall  F1-Score
T-shirt/top          85%      84%      85%
Trouser              99%      97%      98%
Pullover             82%      80%      81%
Dress                89%      91%      90%
Coat                 81%      82%      81%
Sandal               99%      94%      96%
Shirt                71%      71%      71%  (Lowest - needs attention)
Sneaker              92%      97%      95%
Bag                  98%      98%      98%
Ankle boot           95%      95%      95%
```

**Business Cost Analysis:**
- Standard Accuracy: 87.88% (baseline model used for cost calculation)
- Total Misclassification Cost: 3,991
- Maximum Possible Cost: 50,000
- Cost-Weighted Accuracy: 92.02%
- Interpretation: Model makes relatively few high-cost errors

## Performance Metrics

### Confusion Matrix Highlights (Deep Network)
**Top 10 Most Confused Pairs:**
1. Shirt → T-shirt/top: 112 errors [HIGH business impact]
2. T-shirt/top → Shirt: 109 errors [HIGH business impact]
3. Pullover → Coat: 91 errors [MEDIUM business impact]
4. Shirt → Pullover: 78 errors [LOW business impact]
5. Pullover → Shirt: 78 errors [LOW business impact]
6. Coat → Pullover: 76 errors [HIGH business impact]
7. Coat → Shirt: 73 errors [LOW business impact]
8. Shirt → Coat: 65 errors [LOW business impact]
9. Ankle boot → Sneaker: 46 errors [MEDIUM business impact]
10. Dress → Coat: 32 errors [LOW business impact]

### Misclassification Analysis
- **Total Test Samples**: 10,000
- **Total Errors (Deep Network)**: 1,104 (11.04% error rate)
- **Average Confidence on Errors**: 0.67
- **Interpretation**: Model is appropriately less confident on difficult cases

## Team Contributions

### Binger Yu
- Implemented baseline and deep network architectures (model.py)
- Developed training pipeline (train.py)
- Implemented data loading and utilities (utils.py)
- Ran all 3 experiments and model comparisons
- Created confusion matrix and misclassification visualizations
- Performed confidence threshold analysis
- Generated all performance metrics and visualizations
- GitHub repository organization and README, requirements.txt, .gitignore
- Generated Jupyter notebook: fashion_classifier.jpynb
- Created report template on OverLeaf, and Wrote Introduction, Methodology and Results sections

### Vibhor Malik
- Independent model validation (achieved 88.34% with similar baseline)
- Performed business analysis and cost-weighted accuracy calculations
- Created cost matrix visualization
- Generated Jupyter notebook: mini4.jpynb
- Contributed to threshold optimization analysis
- Wrote Business Analysis and Recommendations sections

## Repository Information

**GitHub**: https://github.com/bing-er/mini-project-4.git

**Course**: COMP 9130 - Mini Project 4  
**Institution**: British Columbia Institute of Technology (BCIT)  
**Program**: Master of Science in Applied Computing  
**Instructor**: Dr. Michal Aibin  
**Date**: February 2026

## License
Academic project for COMP 9130 - Mini Project 4

## Acknowledgments
- **Fashion-MNIST Dataset**: Zalando Research ([GitHub](https://github.com/zalandoresearch/fashion-mnist))
- **PyTorch Framework**: Meta AI
- **Course Instructors and TAs**: For guidance and support
- **StyleSort Case Study**: Provided by course materials

## Citation
If you use this work, please cite:
```
Yu, B., & Malik, V. (2026). Fashion-MNIST Classifier for StyleSort: 
Deep Learning Approach to Reduce Product Misclassification. 
COMP 9130 Mini Project 4, BCIT.
```

---

**Note**: The `data/` folder is not included in the repository (.gitignored) as it contains 30MB+ of images. The Fashion-MNIST dataset will be automatically downloaded when you first run the notebook or training scripts.
