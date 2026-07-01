# 07 — Classical ML

[← Sequence Models](06-sequence-models.md) | [Index](README.md) | [Next: GPU Acceleration →](08-gpu-acceleration.md)

The `ml` package is a self-contained scikit-learn-style toolkit: it operates
on plain `[][]float64` / `[]int` slices, has no dependency on the tensor or
autograd machinery, and configures estimators with struct literals (zero
values select sensible defaults).

```go
import "gonn/ml"
```

## 1. The estimator pattern

Every estimator is a struct you configure by fields, then `Fit`/`Predict`:

```go
km := &ml.KMeans{K: 3, MaxIter: 100, Seed: 1} // unset fields -> defaults
km.Fit(X)               // X: [][]float64
labels := km.Predict(X) // []int
```

## 2. Regression

```go
// Ordinary least squares.
lr := &ml.LinearRegression{}
lr.Fit(X, y) // X: [][]float64, y: []float64
yhat := lr.Predict(X)
fmt.Println(lr.Weights, lr.Bias)

// Regularized variants: ml.Ridge, ml.Lasso, ml.ElasticNet, ml.BayesianRidge.
// Boosted trees:
gb := &ml.GradientBoostingRegressor{NEstimators: 100, LR: 0.1, MaxDepth: 3}
gb.Fit(Xtr, ytr)
```

## 3. Classification

```go
// Logistic regression (binary or multiclass).
logit := &ml.LogisticRegression{LR: 0.1, MaxIter: 500}
logit.Fit(X, y) // y: []int class labels
pred := logit.Predict(X)

// Trees and ensembles.
tree := &ml.DecisionTreeClassifier{MaxDepth: 8}
rf := &ml.RandomForestClassifier{NEstimators: 100, MaxDepth: 10, Seed: 42}
rf.Fit(Xtr, ytr)

// Also available: ml.ExtraTreesClassifier, ml.AdaBoostClassifier,
// ml.GradientBoostingClassifier, ml.LinearSVC, ml.KNNClassifier,
// ml.GaussianNB / MultinomialNB / BernoulliNB,
// ml.LinearDiscriminantAnalysis / QuadraticDiscriminantAnalysis.
```

## 4. Clustering & anomaly detection

```go
km := &ml.KMeans{K: 5}                        // k-means++ init
db := &ml.DBSCAN{Eps: 0.3, MinSamples: 5}     // density clustering
gmm := &ml.GaussianMixture{NComponents: 3}    // EM
ms := &ml.MeanShift{Bandwidth: 1.0}
iso := &ml.IsolationForest{NEstimators: 100}  // anomaly scores
```

## 5. Dimensionality reduction & preprocessing

```go
pca := &ml.PCA{NComponents: 2}
pca.Fit(X)
X2 := pca.Transform(X)
fmt.Println(pca.ExplainedVariance)

// Also: ml.KernelPCA, ml.FastICA, ml.TSNE.

scaler := &ml.StandardScaler{}
scaler.Fit(Xtr)
XtrS := scaler.Transform(Xtr)
XteS := scaler.Transform(Xte) // fit on train, apply to test
// Also: ml.MinMaxScaler, ml.OneHotEncoder, ml.LabelEncoder, ml.PolynomialFeatures.
```

## 6. Model selection & metrics

```go
Xtr, Xte, ytrI, yteI := ml.TrainTestSplit(X, y, 0.25, 42)
ytr, yte := ytrI.([]int), yteI.([]int) // y passes through as interface{}

rf := &ml.RandomForestClassifier{NEstimators: 50, MaxDepth: 8, Seed: 1}
rf.Fit(Xtr, ytr)
acc := ml.Accuracy(yte, rf.Predict(Xte))
fmt.Printf("accuracy: %.3f\n", acc)

// Precision / Recall / F1, ConfusionMatrix, MSE / MAE / R2,
// SilhouetteScore, ROCAUC, KFold, CrossValScore are all available.
```

## 7. End-to-end example

[`examples/ml_classical`](../../examples/ml_classical) runs linear
regression, k-means, and PCA in ~50 lines:

```bash
go run ./examples/ml_classical
```

## When to use `ml` vs `nn`

| | `ml` | `nn` + `optim` |
|--|------|----------------|
| Data shape | `[][]float64` slices | `*tensor.Tensor` |
| Fit | closed-form / specialized loops | autograd + SGD-family |
| Sweet spot | tabular data, baselines, clustering, quick experiments | anything with gradients: deep nets, custom losses, GPU training |

They compose naturally: scale features with `ml.StandardScaler`, baseline
with `ml.RandomForestClassifier`, then reach for `nn` when the baseline
stops being enough.

---

[← Sequence Models](06-sequence-models.md) | [Index](README.md) | [Next: GPU Acceleration →](08-gpu-acceleration.md)
