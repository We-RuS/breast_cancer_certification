# Model Comparison Report

| Model               |   Test Score |   CV Accuracy | Best parameters                                                                          |
|---------------------|--------------|---------------|------------------------------------------------------------------------------------------|
| logistic_regression |     0.966667 |      0.97487  | {'C': 0.1, 'solver': 'liblinear'}                                                        |
| knn                 |     0.941667 |      0.97474  | {'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'uniform'}                          |
| swm_linear          |     0.966667 |      0.971169 | {'C': 0.1}                                                                               |
| random_forest       |     0.933333 |      0.967662 | {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 100}                        |
| swm_rbf             |     0.966667 |      0.967597 | {'C': 100, 'gamma': 0.001}                                                               |
| decision_tree       |     0.941667 |      0.942532 | {'criterion': 'entropy', 'max_depth': 30, 'min_samples_leaf': 2, 'min_samples_split': 5} |

## Summary
- **Best Model**: logistic_regression
- **Best CV Score**: 0.9749
- **Test Score**: 0.9667
