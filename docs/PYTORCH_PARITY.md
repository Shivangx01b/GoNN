# PyTorch `torch.nn` Parity Matrix

Coverage of the `torch.nn` catalogue (as of PyTorch 2.x docs) in GoNN.
Legend: ✅ implemented with PyTorch semantics · 🟡 implemented as a documented
Go adaptation · ❌ not implemented (reason given). Every ✅/🟡 item is covered
by finite-difference gradchecks and/or hand-computed value tests; GPU support
means the op lowers to the backend GEMM/kernel dispatch on `-tags cuda` builds.

## Containers & module system

| torch.nn | GoNN | Notes |
|---|---|---|
| Module | ✅ `nn.Base` + `Module`/`Child` | registration-based; `Train()`/`Eval()` propagate |
| Sequential | ✅ `nn.Sequential` | routes children through the hook pipeline |
| ModuleList / ModuleDict | ✅ `nn.ModuleList` / `nn.ModuleDict` | 🟡 Dict iterates keys sorted (Go maps are unordered) |
| ParameterList / ParameterDict | ✅ | |
| Parameter / Buffer | ✅ `RegisterParam` / `RegisterBuffer` | |
| Uninitialized{Parameter,Buffer} | 🟡 via lazy modules | see Lazy section |
| Global + per-module hooks | 🟡 `nn/hooks.go` | forward pre/post + full-backward hooks fire via `nn.Call` and containers; bare `layer.Forward(x)` bypasses them (Go has no `__call__`); backward hook sees `grad_output` only |

## Convolution

| torch.nn | GoNN | Notes |
|---|---|---|
| Conv1d/2d/3d | ✅ | options: `WithStride/WithPad/WithDilation/WithKernel/WithGroups/WithNoBias/WithPaddingMode` (zeros/reflect/replicate/circular — bitwise-equal to manual pre-padding) |
| ConvTranspose1d/2d/3d | ✅ | + `WithOutputPadding` (must be < stride) |
| LazyConv*/LazyConvTranspose* | ✅ `nn.LazyConv…` | run a dry forward before creating the optimizer |
| Unfold / Fold | ✅ `nn.Unfold` / `nn.Fold` | `Fold(Unfold(x)) == x · divisor` identity tested |

## Pooling

| torch.nn | GoNN | Notes |
|---|---|---|
| MaxPool1d/2d/3d | ✅ | + `ForwardWithIndices` (PyTorch flat-index convention), `WithPoolPadding` (−1e300 sentinel, not −Inf — documented), `WithPoolDilation`, `WithPoolCeilMode` (ATen output-clip rule) |
| AvgPool options | ✅ | `WithPoolPadding`, `WithPoolCeilMode` with ATen-exact divisor rules, `WithCountIncludePad(false)` |
| MaxUnpool1d/2d/3d | ✅ | differentiable scatter via `MakeNode` |
| AvgPool1d/2d/3d | ✅ | |
| FractionalMaxPool2d/3d | ✅ | PyTorch `generate_intervals` formula; 🟡 one random u per dim per forward (broadcast `_random_samples`); injectable for determinism |
| LPPool1d/2d/3d | ✅ | sum-then-root, PyTorch-exact |
| AdaptiveMax/AvgPool1d/2d/3d | ✅ | |

## Padding

| torch.nn | GoNN | Notes |
|---|---|---|
| Zero/Constant/Reflection/ReplicationPad 1d/2d/3d | ✅ | 🟡 legacy 2D constructors keep (top,bottom,left,right) order; 1d/3d use PyTorch tuple order |
| CircularPad1d/2d/3d | ✅ | pad ≤ input size enforced |

## Activations

All 25 weighted-sum activations ✅ (`ReLU`…`GLU`), including: `GELU` tanh
**and** exact-erf (`GELUExact()`, `GELUApprox("none"|"tanh")` — with its own
CUDA/OpenCL kernel), `Softplus` with β/threshold (`SoftplusWith`), learnable
`PReLU`, `MultiheadAttention` with additive attention masks, key-padding
masks, attention dropout (`ForwardMasked`), and `WithKDim`/`WithVDim`
projections.
🟡 `RReLU`: training mode samples one slope per forward (whole-tensor, not
per-element); eval = midpoint, PyTorch-exact.

Other: `Softmin`/`Softmax`/`Softmax2d`/`LogSoftmax` ✅,
`AdaptiveLogSoftmaxWithLoss` ✅ (head + per-cluster tails, `WithDivValue`).

## Normalization

| torch.nn | GoNN | Notes |
|---|---|---|
| BatchNorm1d/2d/3d | ✅ | running stats as buffers; Bessel-corrected running var |
| LazyBatchNorm1d/2d/3d | ✅ | |
| GroupNorm / LayerNorm / RMSNorm | ✅ | `WithEps/WithMomentum/WithAffine` |
| InstanceNorm1d/2d/3d (+Lazy) | ✅ | affine off by default (PyTorch parity) |
| LocalResponseNorm | ✅ | exact ATen window semantics |
| SyncBatchNorm | ✅ `distributed.NewSyncBatchNorm1d/2d/3d` | **exact**: stats all-reduced in forward, reduction terms all-reduced inside backward (`MakeNode`); 2-rank split == full-batch BN in values (1e-12) and gradients (1e-10) — tested |

## Recurrent

| torch.nn | GoNN | Notes |
|---|---|---|
| RNN / LSTM / GRU | ✅ | multi-layer + bidirectional; `WithReLU()`; `WithRNNDropout(p)` (inter-layer); LSTM `WithProjSize(p)`; `ForwardWithState(h0…) → (seq, hN[, cN])` in PyTorch `(L·D, B, H)` layout |
| RNNCell / LSTMCell / GRUCell | ✅ | LSTMCell supports projection |
| utils.rnn pack/pad | 🟡 `PackedSequence`, `Pack(Padded)Sequence`, `PadPackedSequence`, `ForwardPacked` | padded storage (semantics-exact, no compute savings); per-sequence outputs/states proven bit-equal to individual runs; bidirectional packed input panics (documented) |

## Transformer

| torch.nn | GoNN | Notes |
|---|---|---|
| Transformer | ✅ `nn.NewTransformer` | batch-first; final encoder/decoder norms like PyTorch |
| TransformerEncoder/Decoder(+Layer) | ✅ | `WithPreNorm` (norm_first), `WithTransformerDropout`, `WithFFActivation`; decoder self-attn causal |

## Linear, Dropout, Sparse, Distance

| torch.nn | GoNN | Notes |
|---|---|---|
| Identity / Linear / Bilinear / LazyLinear | ✅ | |
| Dropout, Dropout1d/2d/3d | ✅ | channel-wise variants zero whole channels |
| AlphaDropout / FeatureAlphaDropout | ✅ | exact ATen α′/a/b constants; mean/var preservation tested |
| Embedding | ✅ | IndexSelect + scatter-add backward; `WithPaddingIdx` (row zeroed, gradient excluded — exactly zero), `WithMaxNorm`/`WithNormType` (in-place renorm, non-differentiable like PyTorch); `scale_grad_by_freq`/`sparse` ❌ |
| EmbeddingBag | ✅ | sum/mean/max; offsets semantics; empty bags = zeros; `WithBagPaddingIdx`/`WithBagMaxNorm` |
| CosineSimilarity / PairwiseDistance | ✅ | |

## Losses — 21/21

`L1`, `MSE`, `CrossEntropy` (✅ class weights, `ignore_index`, label
smoothing — ATen-exact denominators), `CTC` (log-space α/β, analytic gradient
verified against brute-force alignment enumeration; mean divides by
`max(target_len,1)`), `NLL`, `PoissonNLL`, `GaussianNLL`, `KLDiv`, `BCE`,
`BCEWithLogits`, `MarginRanking`, `HingeEmbedding`, `MultiLabelMargin`
(reproduces the PyTorch doc example), `Huber`, `SmoothL1`, `SoftMargin`,
`MultiLabelSoftMargin`, `CosineEmbedding`, `MultiMargin`, `TripletMargin`,
`TripletMarginWithDistance` (custom distance closure). All support
`WithReduction(mean|sum|none)`.

## Vision & shuffle

| torch.nn | GoNN | Notes |
|---|---|---|
| PixelShuffle / PixelUnshuffle | ✅ | |
| Upsample | ✅ | 3D/4D/5D; nearest/linear/bilinear/trilinear/**bicubic** (Keys a=−0.75, verified against PyTorch 2.7.1 float64 goldens at <1e-12) + `WithAlignCorners` |
| UpsamplingNearest2d / UpsamplingBilinear2d | ✅ | bilinear alias uses align_corners=true per PyTorch |
| ChannelShuffle | ✅ | |

## Parallel & distributed

| torch.nn | GoNN | Notes |
|---|---|---|
| DataParallel | 🟡 `nn/parallel.DataParallel` | multi-core **CPU** data parallelism (goroutines over batch shards; grads sum into shared leaves under a leaf-only lock; equals full-batch grads to 1e-12 — tested). BN running stats not replicated (use GroupNorm). Single-GPU accel is orthogonal (backend GEMM). |
| DistributedDataParallel | 🟡 `distributed.Group` | TCP star all-reduce (`AllReduceMeanGrads`, `BroadcastParams`, `Barrier`); 2-rank training keeps params bit-identical across ranks and matches single-process to 1e-12 (tested over localhost) |

## Utilities

| torch.nn.utils | GoNN | Notes |
|---|---|---|
| clip_grad_norm_ / clip_grad_value_ | ✅ `optim.ClipGradNorm` / `ClipGradValue` | |
| get_total_norm / clip_grads_with_norm_ | ✅ `optim.TotalGradNorm` / `ClipGradsWithNorm` | |
| parameters_to_vector / vector_to_parameters | ✅ `optim.ParametersToVector` / `VectorToParameters` (+`GradsToVector`) | |
| fuse_conv_bn_eval / fuse_linear_bn_eval | ✅ `nn.FuseConvBNEval` / `FuseLinearBNEval` | equivalence verified to 1e-12 |
| weight_norm / spectral_norm | 🟡 `nn.WeightNormLinear/Conv2d`, `nn.SpectralNormLinear/Conv2d` | explicit wrapper modules (no hooks); power-iteration buffers update in train mode; `Remove*` bakes weights back |
| prune.* | 🟡 `nn/prune` | mask registry + `Reapply` after optimizer steps replaces the forward pre-hook; Random/L1 unstructured, Ln/Random structured, global, custom-mask, remove/is-pruned |
| parametrize.* | 🟡 `nn.Parametrization` + `ParametrizedLinear/Conv2d` | generic function-of-weight wrappers with chaining (`AddParametrization`) and `RemoveParametrizations`; no `right_inverse`/`cached` |
| stateless.functional_call | 🟡 `nn.FunctionalCall` / `FunctionalCallGrad` | temporary param-data swap; gradients land on module params (use `FunctionalCallGrad` for grads-by-name — sharp edges documented) |
| skip_init | ❌ | intentionally N/A: no meta device; constructor init cost is negligible in Go |

## Quantization

🟡 `quant` package: int8 per-tensor **and per-channel** affine (`QTensor`,
`QTensorPerChannel`, `Quantize[PerChannel]`, observers incl.
`MovingAverageMinMaxObserver`) with **dynamic** and **static** quantized
Linear (`WithPerChannelWeights` option) — int8×int8→int32 GEMM with
zero-point correction. **QAT**: `FakeQuant` (clamped straight-through
gradients via `MakeNode`), `QATLinear` → `Convert()` → `StaticLinear`
(converted output matches QAT eval to 1.4e-14; QAT beats naive PTQ ~10× on a
bias-compensable fixture — tested). CPU-only, inference-only (PyTorch eager
quantization is likewise CPU). Quantized conv ❌.

## Lazy modules

✅ 13 modules: `LazyLinear`, `LazyConv{,Transpose}1d/2d/3d`,
`LazyBatchNorm1d/2d/3d`, `LazyInstanceNorm1d/2d/3d` + `IsInitialized()` /
`nn.IsLazy()`. 🟡 As in PyTorch, run one dry forward before constructing the
optimizer (`Parameters()` is empty pre-initialization — documented loudly).

## Known remaining gaps (complete list)

Hooks on bare `Forward` calls (Go has no `__call__` — use `nn.Call` or
containers) · `skip_init` (N/A — no meta device) · bidirectional
`ForwardPacked` (padded storage can't reproduce the packed reverse pass —
panics with guidance) · `Embedding` `scale_grad_by_freq`/`sparse` gradients ·
quantized conv / fx-graph quantization · `parametrize` `right_inverse`/
`cached` · `FractionalMaxPool` per-(n,c) random samples (one draw per dim,
broadcast).
