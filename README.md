# Fashion-MNIST Classifier for StyleSort

## Team Members - Group 8
- Binger Yu
- Vibhor Malik

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
| Baseline | 784→128→10 | 87.80% | Simple 2-layer network |
| Deep Network | 784→256→128→64→10 | **88.12%** | Dropout + BatchNorm |
| Lower LR | 784→128→10 | 85.24% | Learning rate 0.0001 |

**Best Model**: Deep Network (88.12% accuracy) ✓ Exceeds 85% requirement

### Business Impact
- **Cost-Weighted Accuracy**: 92.20%
  - Model errors cost only 7.8% of maximum possible misclassification cost
- **Recommended Confidence Threshold**: 0.80
  - 95.84% accuracy on accepted predictions
  - Auto-processes 78% of items (7,800/day)
  - 22% flagged for human review (2,200/day)

### Key Insights
1. **Most Confused Pairs**:
   - Shirt ↔ T-shirt (135 errors) - High business cost
   - Pullover ↔ Coat (115 errors) - High business cost
   - Shirt ↔ Pullover (97 errors)

2. **Best Performing Categories**: Trouser (99%), Bag (97%), Sandal (98%)
3. **Challenging Categories**: Shirt (67% precision) - needs improved photography guidelines

## Business Recommendations

### Immediate Actions
1. **Deploy with 0.80 confidence threshold**
   - Automatically categorize 78% of items with 95.84% accuracy
   - Route 22% to human reviewers for quality assurance

2. **Improve Product Photography for Confused Categories**
   - Shirt vs. T-shirt: Emphasize collar, buttons, formality
   - Pullover vs. Coat: Show fabric weight, lining, intended use
   - Provide photography guidelines to vendors

3. **Enhanced Descriptions for High-Cost Pairs**
   - Add "Style: Casual/Formal" tags
   - Include "Warmth Rating" for outerwear
   - Specify "Season: Summer/Winter/All-Season"

### Future Improvements
- Collect more training data for Shirt category
- Implement data augmentation (rotation, brightness)
- Explore CNN architectures for better feature extraction
- A/B test confidence thresholds in production

## Setup Instructions

### Requirements
```bash
pip install -r requirements.txt
```

### Running the Project

**Option 1: Jupyter Notebook** (Recommended)
```bash
jupyter notebook notebooks/fashion_classifier.ipynb
```

**Option 2: Python Scripts**
```bash
# Train a model
python src/train.py

# Or import and use
from src.model import FashionClassifier
from src.train import train_model
```

### Project Structure
```
mini-project-4/
├── README.md
├── requirements.txt
├── notebooks/
│   └── fashion_classifier.ipynb    # Main analysis notebook
├── src/
│   ├── model.py                     # Neural network architectures
│   ├── train.py                     # Training loops and evaluation
│   └── utils.py                     # Data loading and helpers
└── results/
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── confidence_threshold_analysis.png
    ├── misclassified_examples.png
    └── experiment_comparison.png
```

## Technical Details

### Model Architecture (Best Performing)
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
```

### Training Configuration
- **Dataset**: Fashion-MNIST (60K train, 10K test)
- **Train/Val Split**: 54K / 6K
- **Batch Size**: 64
- **Epochs**: 15 (Deep Network), 10 (Baseline)
- **Hardware**: CPU (training time ~5 minutes per model)

## Team Contributions

### Binger Yu
- Implemented baseline and deep network architectures (model.py)
- Developed training pipeline (train.py)
- Ran experiments 1 and 2
- Created confusion matrix and misclassification visualizations
- Wrote Methodology and Results sections

### Vibhor Malik
- Implemented data loading and utilities (utils.py)
- Ran experiment 3
- Performed business analysis (cost-weighted accuracy, threshold analysis)
- Created confidence threshold visualizations
- Wrote Business Analysis and Recommendations sections

### Joint Work
- Project setup and environment configuration
- Model architecture design discussions
- Results interpretation and business recommendations
- Final report and documentation review

## License
Academic project for COMP 9130 - Mini Project 4

## Acknowledgments
- Fashion-MNIST dataset: Zalando Research
- PyTorch framework: Meta AI
- Course instructors and TAs