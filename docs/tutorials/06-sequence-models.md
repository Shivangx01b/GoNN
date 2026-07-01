# 06 — Sequence Models

[← Convolutional Networks](05-convolutional-networks.md) | [Index](README.md) | [Next: Classical ML →](07-classical-ml.md)

Recurrent networks, sequence-to-sequence, attention, and transformers. All
sequence layers use batch-first layout: `(B, T, features)`.

## 1. RNN, LSTM, GRU

One constructor shape covers single-layer through deep bidirectional stacks:

```go
rnn := nn.NewRNN(inputSize, hiddenSize)                    // 1 layer, unidirectional
lstm := nn.NewLSTM(64, 128, nn.WithLayers(2))              // 2 stacked layers
gru := nn.NewGRU(64, 128, nn.WithLayers(2), nn.WithBidirectional())

x := tensor.Randn(8, 20, 64)  // (batch, time, features)
h := lstm.Forward(x)          // (8, 20, 128) — hidden states for every step
hb := gru.Forward(x)          // (8, 20, 256) — fwd ++ bwd features
```

The output is the full hidden-state sequence. Common heads:

```go
// Last time step -> classification:
last := h.IndexSelect(1, tensor.Scalar(19).Reshape(1)).Reshape(8, 128)
logits := head.Forward(last)

// Every time step -> per-token prediction (Linear maps the last dim):
perStep := nn.NewLinear(128, vocab, true).Forward(h) // (8, 20, vocab)
```

## 2. Cells: driving the recurrence yourself

`RNNCell`, `LSTMCell`, `GRUCell` are the single-timestep primitives — the
building blocks for custom decoding loops (sampling, beam search, scheduled
sampling):

```go
cell := nn.NewLSTMCell(embed, hidden)
var state *nn.LSTMState // nil = zero state
for t := 0; t < T; t++ {
	state = cell.Forward(xt(t), state) // state.H, state.C: (B, hidden)
}
```

## 3. Seq2Seq

A minimal teacher-forced encoder–decoder (embedding → LSTM encoder → LSTM
decoder → vocab projection):

```go
m := nn.NewSeq2Seq(srcVocab, tgtVocab, embedDim, hidden)

src := tensor.New([]float64{1, 2, 3, 0, 4, 5}, 2, 3) // (B, T_src) token ids
tgt := tensor.New([]float64{0, 1, 2, 3}, 2, 2)       // (B, T_tgt) decoder inputs
logits := m.Forward(src, tgt)                        // (B, T_tgt, tgtVocab)

// Train against the shifted targets with CrossEntropyLoss over the
// flattened (B*T, vocab) logits.
```

## 4. Multi-head attention

```go
mha := nn.NewMultiHeadAttention(512, 8) // embedDim, numHeads

// Self-attention with a causal mask (each position sees only its past):
y := mha.Forward(x, x, x, true)

// Cross-attention (decoder queries attend over encoder memory):
y = mha.Forward(queries, memory, memory, false)
```

Internally the scaled dot-product core is two **batched matmuls**
(`Q.MatMul(K.Transpose())` over `(B, H, T, D)` tensors) — one strided-batched
GEMM each, so attention rides the GPU GEMM path on CUDA builds. On CUDA
builds with `headDim == 64`, `Forward` automatically routes through the
**fused flash-attention kernel** — forward *and* backward — so the same
training code gets the fused speedup with no changes (tutorial 08).

## 5. Transformers

Post-norm encoder/decoder blocks, PyTorch-style:

```go
enc := nn.NewTransformerEncoder(6, 512, 8, 2048)       // layers, embed, heads, dimFF
memory := enc.Forward(src)                             // (B, T_src, 512)

dec := nn.NewTransformerDecoder(6, 512, 8, 2048)
out := dec.Forward(tgt, memory)                        // causal self-attn + cross-attn
```

A small but complete text classifier (embedding + encoder + mean pool):

```go
type TextClassifier struct {
	nn.Base
	Emb  *nn.Embedding
	Enc  *nn.TransformerEncoder
	Head *nn.Linear
}

func NewTextClassifier(vocab, embed, heads, ff, layers, classes int) *TextClassifier {
	m := &TextClassifier{
		Emb:  nn.NewEmbedding(vocab, embed),
		Enc:  nn.NewTransformerEncoder(layers, embed, heads, ff),
		Head: nn.NewLinear(embed, classes, true),
	}
	m.RegisterChild("emb", m.Emb)
	m.RegisterChild("enc", m.Enc)
	m.RegisterChild("head", m.Head)
	return m
}

func (m *TextClassifier) Forward(tokens *tensor.Tensor) *tensor.Tensor {
	h := m.Enc.Forward(m.Emb.Forward(tokens)) // (B, T, E)
	pooled := h.MeanAxis(1, false)            // (B, E)
	return m.Head.Forward(pooled)             // (B, classes)
}
```

[`examples/transformer`](../../examples/transformer) trains exactly this
shape of model to 100% on a synthetic task.

## 6. Practical notes

- **Gradient clipping matters here.** Recurrent nets are the textbook case
  for exploding gradients — `optim.ClipGradNorm(opt.Parameters(), 1.0)`
  between `Backward()` and `Step()` (tutorial 04 §4).
- **Deep unrolls are safe.** The autograd engine sorts the graph
  iteratively, so a 100k-op unrolled sequence cannot overflow the stack.
- **Embedding is O(tokens)**: lookups use `IndexSelect` with a scatter-add
  backward, not a one-hot matmul, so a large vocabulary costs what it should.

---

[← Convolutional Networks](05-convolutional-networks.md) | [Index](README.md) | [Next: Classical ML →](07-classical-ml.md)
