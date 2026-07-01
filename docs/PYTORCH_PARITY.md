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
| Conv1d/2d/3d | ✅ | options: `WithStride/WithPad/WithDilation/WithKernel/WithGroups/WithNoBias`; `padding_mode` ❌ (compose with padding layers) |
| ConvTranspose1d/2d/3d | ✅ | + `WithOutputPadding` (must be < stride) |
| LazyConv*/LazyConvTranspose* | ✅ `nn.LazyConv…` | run a dry forward before creating the optimizer |
| Unfold / Fold | ✅ `nn.Unfold` / `nn.Fold` | `Fold(Unfold(x)) == x · divisor` identity tested |

## Pooling

| torch.nn | GoNN | Notes |
|---|---|---|
| MaxPool1d/2d/3d | ✅ | + `ForwardWithIndices` (PyTorch flat-index convention); pool `padding`/`dilation`/`ceil_mode` ❌ |
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
masks, and attention dropout (`ForwardMasked`; `kdim`/`vdim` ❌).
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
| SyncBatchNorm | ❌ | BN-stat sync across ranks not implemented — use GroupNorm with DDP (documented in `distributed`) |

## Recurrent

| torch.nn | GoNN | Notes |
|---|---|---|
| RNN / LSTM / GRU | ✅ | multi-layer + bidirectional; `WithReLU()` nonlinearity; `ForwardWithState(h0…) → (seq, hN[, cN])` in PyTorch `(L·D, B, H)` layout; inter-layer dropout ❌, `proj_size` ❌ |
| RNNCell / LSTMCell / GRUCell | ✅ | |
| utils.rnn pad_sequence / unpad_sequence | ✅ `nn.PadSequence`/`UnpadSequence` | `PackedSequence` ❌ — GoNN RNNs consume padded batches (documented) |

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
| Embedding | ✅ | IndexSelect + scatter-add backward; `padding_idx`/`max_norm` ❌ |
| EmbeddingBag | ✅ | sum/mean/max; offsets semantics; empty bags = zeros |
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
| Upsample | ✅ | 3D/4D/5D; nearest/linear/bilinear/trilinear + `WithAlignCorners`; bicubic ❌ |
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
| parametrize.* | 🟡 | covered by the weight/spectral-norm wrappers; the generic register_parametrization framework ❌ |
| skip_init / stateless.functional_call | ❌ | not meaningful without reflection-driven construction |

## Quantization

🟡 `quant` package: int8 per-tensor affine (`QTensor`, `Quantize`/`Dequantize`,
`MinMaxObserver`) with **dynamic** and **static** quantized Linear
(`NewDynamicLinearFrom`, `NewStaticLinearFrom`) — int8×int8→int32 GEMM with
zero-point correction, ~1% relative error on well-scaled inputs (tested).
CPU-only, inference-only (PyTorch eager quantization is likewise CPU).
Fake-quant QAT, per-channel schemes, and quantized conv ❌.

## Lazy modules

✅ 13 modules: `LazyLinear`, `LazyConv{,Transpose}1d/2d/3d`,
`LazyBatchNorm1d/2d/3d`, `LazyInstanceNorm1d/2d/3d` + `IsInitialized()` /
`nn.IsLazy()`. 🟡 As in PyTorch, run one dry forward before constructing the
optimizer (`Parameters()` is empty pre-initialization — documented loudly).

## Known remaining gaps (complete list)

`padding_mode` on convs · pool `padding`/`ceil_mode`/`dilation` · MHA
`kdim`/`vdim` · RNN inter-layer dropout, `proj_size`, `PackedSequence` ·
`SyncBatchNorm` · bicubic upsampling · `Embedding` `padding_idx`/`max_norm` ·
generic `register_parametrization` · QAT/per-channel quantization ·
`skip_init`/`stateless` · hooks on bare `Forward` calls.
