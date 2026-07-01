# 04 — Training

[← Building Models](03-building-models.md) | [Index](README.md) | [Next: Convolutional Networks →](05-convolutional-networks.md)

Losses, optimizers, parameter groups, gradient clipping, LR schedulers, and
batched data loading — assembled into a production-shaped training loop.

## 1. Losses

All losses are plain functions returning a scalar tensor (mean reduction by
default):

```go
loss := nn.MSELoss(pred, target)
loss := nn.CrossEntropyLoss(logits, classIdx) // logits (N,C); targets (N,) as float64 class ids
loss := nn.BCEWithLogitsLoss(logits, binary)  // numerically stable
loss := nn.HuberLoss(pred, target, 1.0)
loss := nn.TripletMarginLoss(anchor, pos, neg, 0.5)
```

Full set: `MSELoss`, `MAELoss`/`L1Loss`, `HuberLoss`/`SmoothL1Loss`,
`CrossEntropyLoss`, `NLLLoss`, `BCELoss`, `BCEWithLogitsLoss`, `KLDivLoss`,
`PoissonNLLLoss`, `GaussianNLLLoss`, `MarginRankingLoss`,
`HingeEmbeddingLoss`, `CosineEmbeddingLoss`, `TripletMarginLoss`,
`MultiMarginLoss`.

Every loss accepts a trailing reduction option:

```go
perSample := nn.CrossEntropyLoss(logits, y, nn.WithReduction(nn.ReduceNone)) // (N,)
total     := nn.MSELoss(pred, y, nn.WithReduction(nn.ReduceSum))             // scalar sum
```

## 2. Optimizers

Sixteen optimizers share one interface — construct with the parameter list
and a learning rate, then `ZeroGrad`/`Step`:

```go
opt := optim.NewAdam(model.Parameters(), 1e-3)
opt := optim.NewSGD(params, 0.1, optim.WithMomentum(0.9), optim.WithNesterov(true))
opt := optim.NewAdamW(params, 3e-4) // decoupled weight decay, default 0.01
```

Available: `SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, NAdam, Adamax,
RAdam, Rprop, Adafactor, ASGD, LAMB, Lion, SparseAdam` — plus `LBFGS`, which
is closure-driven:

```go
lbfgs := optim.NewLBFGS(params, 1.0)
for i := 0; i < 20; i++ {
	lbfgs.Step(func() float64 { // re-evaluates the loss during line search
		lbfgs.ZeroGrad()
		loss := nn.MSELoss(model.Forward(X), Y)
		loss.Backward()
		return loss.Item()
	})
}
```

## 3. Parameter groups

Give different parts of the model different learning rates or weight decay —
the classic backbone/head split, or "no decay on biases":

```go
opt := optim.NewAdamWGroups([]optim.Group{
	{Params: backbone.Parameters(), LR: 1e-4, WeightDecay: 0.01},
	{Params: head.Parameters(), LR: 1e-3}, // 10x LR, no decay
})

// Build groups by name with the nn helpers:
decay := nn.FilterParams(model, func(n string) bool { return strings.HasSuffix(n, ".weight") })
noDecay := nn.FilterParams(model, func(n string) bool { return !strings.HasSuffix(n, ".weight") })
opt = optim.NewAdamWGroups([]optim.Group{
	{Params: decay, LR: 3e-4, WeightDecay: 0.01},
	{Params: noDecay, LR: 3e-4},
})

// Groups are live — mutate them mid-training:
opt.Groups()[0].LR = 1e-5 // e.g. freeze-ish the backbone for fine-tuning
```

`LR: 0` legitimately freezes a group. `opt.SetLR(x)` sets **every** group
(the classic single-group behavior); use `Groups()` for per-group control.

## 4. Gradient clipping

Free functions, called between `Backward()` and `Step()`:

```go
loss.Backward()
norm := optim.ClipGradNorm(opt.Parameters(), 1.0) // global L2 clip; returns pre-clip norm
opt.Step()

// or elementwise clamping:
optim.ClipGradValue(opt.Parameters(), 0.5)
```

The returned pre-clip norm is the number you want on your metrics dashboard —
a spiking gradient norm is the classic early warning of divergence.

## 5. LR schedulers

Schedulers wrap the optimizer and adjust learning rates on `Step()`. With
parameter groups, each group is scheduled **relative to its own base LR**, so
your backbone/head ratio survives the schedule:

```go
sched := optim.NewCosineAnnealingLR(opt, 1000, 0)   // TMax steps, etaMin
sched := optim.NewStepLR(opt, 30, 0.1)              // ×0.1 every 30 steps
sched := optim.NewOneCycleLR(opt, 1e-2, totalSteps) // ramp to maxLR and back
sched := optim.NewLinearLR(opt, 0.1, 1.0, 500)      // warmup: 10% -> 100% over 500 steps

for step := 0; step < totalSteps; step++ {
	// ... ZeroGrad / forward / Backward / opt.Step() ...
	sched.Step()
}
```

`ReduceLROnPlateau` is metric-driven — it takes the observed loss:

```go
plateau := optim.NewReduceLROnPlateau(opt, 0.1, 1e-4, 10) // factor, threshold, patience
// each validation epoch:
plateau.Step(valLoss)
```

Compose schedules with `ChainedScheduler` (run several per step) and
`SequentialLR` (switch at milestones — e.g. linear warmup then cosine decay).

## 6. Batched data with DataLoader

```go
X, Y := data.MakeClassification(1000, 20, 3, 42) // samples, features, classes, seed
ds := data.NewTensorDataset(X, Y)
loader := data.NewDataLoader(ds, 32, true) // batch size 32, shuffle

for batch := range loader.Iter() {
	// batch.X: (32, 20), batch.Y: (32,)
}
```

Synthetic generators for experiments: `MakeRegression`, `MakeClassification`,
`MakeBlobs`, `MakeMoons`. Real data: `data.LoadCSV`, MNIST helpers, and
composable transforms (`Normalize`, `RandomHorizontalFlip`, `RandomCrop`).

## 7. Putting it together

A complete, runnable classifier with everything from this tutorial:

```go
package main

import (
	"fmt"

	"gonn/data"
	"gonn/nn"
	"gonn/optim"
)

func main() {
	X, Y := data.MakeClassification(600, 10, 3, 7)
	loader := data.NewDataLoader(data.NewTensorDataset(X, Y), 64, true)

	model := nn.NewSequential(
		nn.NewLinear(10, 64, true),
		nn.GELU(),
		nn.NewDropout(0.1),
		nn.NewLinear(64, 3, true),
	)

	opt := optim.NewAdamW(model.Parameters(), 3e-3)
	sched := optim.NewCosineAnnealingLR(opt, 200, 1e-5)

	model.Train()
	for epoch := 0; epoch < 20; epoch++ {
		var epochLoss float64
		var batches int
		for batch := range loader.Iter() {
			opt.ZeroGrad()
			logits := model.Forward(batch.X)
			loss := nn.CrossEntropyLoss(logits, batch.Y)
			loss.Backward()
			optim.ClipGradNorm(opt.Parameters(), 1.0)
			opt.Step()
			sched.Step()
			epochLoss += loss.Item()
			batches++
		}
		if epoch%5 == 0 || epoch == 19 {
			fmt.Printf("epoch %2d  loss=%.4f  lr=%.5f\n", epoch, epochLoss/float64(batches), opt.LR())
		}
	}

	// Evaluation: dropout off.
	model.Eval()
	logits := model.Forward(X)
	correct := 0
	pred := logits.ArgMax(1)
	for i := range pred.Data {
		if int(pred.Data[i]) == int(Y.Data[i]) {
			correct++
		}
	}
	fmt.Printf("train accuracy: %.1f%%\n", 100*float64(correct)/float64(len(pred.Data)))
}
```

---

[← Building Models](03-building-models.md) | [Index](README.md) | [Next: Convolutional Networks →](05-convolutional-networks.md)
