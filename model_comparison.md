# Comprehensive Comparison of Classification Models for Legal Documents

This document provides a detailed comparison of the four machine learning models implemented for classifying EU legal documents (Regulations, Directives, and Decisions), as well as a reference comparison to the BERT model.

## 1. Performance Metrics

### Accuracy

| Model | Accuracy | Training Set Size | Test Set Size |
|-------|----------|-------------------|---------------|
| Logistic Regression | 99.62% | 36,000 | 9,000 |
| SVM | 99.91% | 36,000 | 9,000 |
| Random Forest | 99.94% | 36,000 | 9,000 |
| Naive Bayes | 93.92% | 36,000 | 9,000 |
| BERT (reference) | ~100% | 36,000 | 9,000 |

### Per-Class Performance (F1 Scores)

| Model | Decision | Directive | Regulation | Macro Avg |
|-------|----------|-----------|------------|-----------|
| Logistic Regression | 0.99 | 0.99 | 1.00 | 0.99 |
| SVM | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest | 1.00 | 0.99 | 1.00 | 1.00 |
| Naive Bayes | 0.94 | 0.85 | 0.95 | 0.91 |
| BERT (reference) | ~1.00 | ~1.00 | ~1.00 | ~1.00 |

## 2. Computational Performance

### Training and Inference Times

| Model | Training Time | Inference Time (per doc) | Parallel Processing |
|-------|---------------|--------------------------|---------------------|
| Logistic Regression | 130.39s | 2.76ms | Yes (n_jobs=-1) |
| SVM | 129.65s | 3.37ms | Limited |
| Random Forest | 131.68s | 3.49ms | Yes (n_jobs=-1) |
| Naive Bayes | 101.61s | 3.32ms | Limited |
| BERT (reference) | Hours | ~250-500ms | GPU acceleration |

### Resource Requirements

| Model | Memory Usage (Training) | Model Size | Scalability | Hardware Requirements |
|-------|-------------------------|------------|-------------|----------------------|
| Logistic Regression | ~0.5-1 GB | ~5-10 MB | Excellent | CPU only, 2+ GB RAM |
| SVM | ~1-2 GB | ~10-20 MB | Good | CPU only, 4+ GB RAM |
| Random Forest | ~2-4 GB | ~10-40 MB | Good | CPU only, 4+ GB RAM |
| Naive Bayes | ~0.5-1 GB | ~5-10 MB | Excellent | CPU only, 2+ GB RAM |
| BERT (reference) | ~8-16 GB | ~438 MB | Limited | GPU recommended, 8+ GB RAM |

## 3. Model Characteristics

### Interpretability

| Model | Interpretability | Explanation Method | Feature Insights |
|-------|------------------|-------------------|------------------|
| Logistic Regression | Very High | Coefficient weights | Direct relationship between features and classes |
| SVM | High | Feature coefficients | Linear separability of features |
| Random Forest | High | Feature importance | Non-linear relationships, feature interactions |
| Naive Bayes | Moderate | Conditional probabilities | Independent feature contributions |
| BERT (reference) | Low | Attention visualizations | Complex, contextual embeddings |

### Top Features by Model

#### Logistic Regression
- **Decision**: "decision" (17.5), "this decision" (12.6), "decision of" (6.4)
- **Directive**: "this directive" (17.5), "directive" (14.9), "directive is" (5.2)
- **Regulation**: "regulation" (14.0), "this regulation" (10.9), "regulation shall" (6.6)

#### SVM
- **Decision**: "this decision" (7.5), "decision" (7.3), "decision of" (4.1)
- **Directive**: "this directive" (7.9), "directive" (4.2), "directive is" (2.7)
- **Regulation**: "this regulation" (7.3), "regulation" (6.5), "regulation shall" (4.3)

#### Random Forest
- **Decision**: "decision" (77.0), "this directive" (67.0), "decision of" (62.0)
- **Directive**: "directive" (28.0), "this directive" (27.0), "directive is" (26.0)
- **Regulation**: "directly applicable" (48.0), "and directly" (47.0), "entirety" (43.0)

#### Naive Bayes
- Feature importance not directly comparable due to different mechanism (probability ratios)

## 4. Deployment Considerations

### Web Deployment Characteristics

| Model | API Response Time | Batch Processing | Concurrent Users Support | Cold Start Time |
|-------|-------------------|------------------|--------------------------|----------------|
| Logistic Regression | ~5-10ms | Excellent | High | Very fast |
| SVM | ~5-15ms | Good | High | Very fast |
| Random Forest | ~10-20ms | Good | Medium-High | Fast |
| Naive Bayes | ~5-10ms | Excellent | High | Very fast |
| BERT (reference) | ~300-600ms | Limited | Low-Medium | Slow |

### Scaling Considerations

| Model | Memory Per Instance | Horizontal Scaling | Model Update Complexity |
|-------|---------------------|---------------------|-------------------------|
| Logistic Regression | Low | Easy | Simple |
| SVM | Low | Easy | Moderate |
| Random Forest | Medium | Moderate | Moderate |
| Naive Bayes | Low | Easy | Simple |
| BERT (reference) | High | Difficult | Complex |

## 5. Use Case Suitability

| Scenario | Best Model | Reasoning |
|----------|------------|-----------|
| High-throughput web API | Logistic Regression | Fastest inference, excellent accuracy, low resource usage |
| Maximum accuracy requirement | Random Forest or SVM | Highest accuracy with reasonable speed |
| Low-powered device deployment | Naive Bayes or Logistic Regression | Lowest resource requirements |
| Need for feature importance analysis | Logistic Regression or Random Forest | Most interpretable feature importance |
| Document processing pipeline | SVM or Logistic Regression | Balance of speed and accuracy |
| Academic/research | BERT | State-of-the-art performance, despite resource requirements |

## 6. Implementation Complexity

| Model | Training Code Complexity | Hyperparameter Tuning | Maintenance Effort |
|-------|--------------------------|------------------------|---------------------|
| Logistic Regression | Low | Low (few hyperparameters) | Low |
| SVM | Low | Moderate (kernel, C, class_weight) | Low |
| Random Forest | Low | High (many hyperparameters) | Moderate |
| Naive Bayes | Very Low | Very Low (minimal tuning needed) | Very Low |
| BERT (reference) | High | Very High (many hyperparameters) | High |

## 7. Summary and Recommendations

### Best Overall Model
**Logistic Regression** offers the best balance of accuracy (99.62%), speed (2.76ms per inference), interpretability, and resource efficiency. It's an excellent choice for production deployment in a web application.

### Best for Highest Accuracy
**Random Forest** (99.94%) and **SVM** (99.91%) offer the highest accuracy, nearly matching BERT's performance with far less computational cost.

### Best for Resource-Constrained Environments
**Naive Bayes** has the fastest training time and lowest memory requirements, though with a significant accuracy trade-off (93.92%).

### Best for Interpretability
**Logistic Regression** provides the most straightforward interpretation of feature importance through its coefficient weights, making it ideal for understanding classification decisions.

### Recommendation for Production Deployment
A tiered approach:
1. **Primary Model**: Logistic Regression for most classifications (fastest, excellent accuracy)
2. **Secondary Model**: SVM for cases where Logistic Regression has low confidence
3. **Tertiary Model**: BERT only for the most difficult cases where traditional models fail

### For Future Work
- Ensemble methods combining multiple models
- Additional feature engineering to improve Naive Bayes performance
- Optimization of Random Forest parameters for faster inference
- Distillation of BERT model knowledge into smaller models
