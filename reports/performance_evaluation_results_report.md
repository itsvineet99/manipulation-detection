# Performance Evaluation and Results Obtained

Generated on: `2026-03-23 20:50:32`

## 1. Evaluation Objective
This report summarizes how the current manipulation detection pipeline performs on the synthetic
training dataset and under held-out cross-validation. The goal is to document both pipeline fit
quality and generalization within the same data generation regime.

## 2. Evaluation Setup
Dataset under evaluation: `data/manipulation_detection_dataset.csv` with 10000 rows. Each row
supplies one text input and supervised targets for `label`, `intent_label`, `manipulation_type`,
`domain`, and `severity`.

Training-fit evaluation: the model is trained on the full dataset and then scored on that same
dataset. This is a useful sanity check for implementation correctness, but it is not the main
estimate of generalization.

Held-out evaluation: `evaluate_model.py` performs 5-fold `StratifiedKFold` cross-validation.
Stratification is done on the combined key `label|intent_label` so that the positive/negative
classes and intent distribution remain stable across folds.

Metrics reported: accuracy, macro F1, and weighted F1 for each target. For the binary label
target, a full classification report is also included.

### Dataset Snapshot
- Vectorized feature space size: `16535` features
- Label balance: `manipulative=5000`, `not_manipulative=5000`

**Label balance chart**
```text
manipulative       [########################] 5000
not_manipulative   [########################] 5000
```

## 3. Training-Fit Results
| Target | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- |
| `label` | 1.0000 | 1.0000 | 1.0000 |
| `intent_label` | 1.0000 | 1.0000 | 1.0000 |
| `manipulation_type` | 1.0000 | 1.0000 | 1.0000 |
| `domain` | 1.0000 | 1.0000 | 1.0000 |
| `severity` | 1.0000 | 1.0000 | 1.0000 |

All training-fit metrics are 1.0000 on the current dataset. This indicates that the pipeline is
internally consistent and that the classifiers can fully separate the generated examples after
vectorization.

## 4. Cross-Validation Results
### Fold-wise Binary Label Accuracy
| Fold | Label Accuracy |
| --- | --- |
| 1 | 1.0000 |
| 2 | 1.0000 |
| 3 | 1.0000 |
| 4 | 1.0000 |
| 5 | 1.0000 |

### Aggregate 5-Fold Metrics
| Target | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- |
| `label` | 1.0000 | 1.0000 | 1.0000 |
| `intent_label` | 1.0000 | 1.0000 | 1.0000 |
| `manipulation_type` | 1.0000 | 1.0000 | 1.0000 |
| `domain` | 1.0000 | 1.0000 | 1.0000 |
| `severity` | 1.0000 | 1.0000 | 1.0000 |

**Cross-validation accuracy chart**
```text
label              [##############################] 1.0000
intent_label       [##############################] 1.0000
manipulation_type  [##############################] 1.0000
domain             [##############################] 1.0000
severity           [##############################] 1.0000
```

### Binary Label Classification Report
```text
                  precision    recall  f1-score   support

    manipulative       1.00      1.00      1.00      5000
not_manipulative       1.00      1.00      1.00      5000

        accuracy                           1.00     10000
       macro avg       1.00      1.00      1.00     10000
    weighted avg       1.00      1.00      1.00     10000
```

## 5. Interpretation
The `label`, `manipulation_type`, and `severity` targets achieve perfect held-out performance
under the current synthetic split. That suggests the language patterns for those targets are
highly separable in the generated dataset.

In the current 10,000-row version, `intent_label` and `domain` also reach perfect cross-
validation scores. This means the expanded synthetic dataset is still highly structured and
cleanly separable across all supervised targets.

Because the evaluation folds come from the same synthetic generation process as the training
data, these results should be interpreted as controlled-benchmark performance rather than final
real-world accuracy.

Even after expanding the dataset to 10,000 rows and adding noisier chat-style surface variation,
the current target space remains highly separable under this synthetic generation regime. That
is why cross-validation remains perfect in this version of the project.

## 6. Main Findings
1. The end-to-end training and inference pipeline is functioning correctly and produces stable outputs across all targets.
2. The shared TF-IDF plus logistic regression approach is sufficient to separate the current synthetic classes with very high accuracy.
3. The added chat-style noise and shorter-message variation were not enough to break separability under the current synthetic generation process.
4. Real-world validation remains the next major step before making stronger claims about deployment robustness.

## 7. Reproducibility
- Training command: `.venv/bin/python train_manipulation_model.py`
- Evaluation command: `.venv/bin/python evaluate_model.py`
- Single inference command: `.venv/bin/python manipulation_inference.py --text "If you cared, you would send the report tonight."`
