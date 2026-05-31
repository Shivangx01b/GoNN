// Tiny transformer encoder demo. Builds a 2-layer encoder over a small
// sequence-classification task: classify the sum of a length-4 sequence
// of one-hot tokens as even or odd.
package main

import (
	"fmt"
	"math/rand"

	"gonn/nn"
	"gonn/optim"
	"gonn/tensor"
)

const (
	vocab   = 8
	seqLen  = 4
	embDim  = 16
	heads   = 4
	dimFF   = 32
	classes = 2
)

func main() {
	rand.Seed(11)

	embedding := nn.NewEmbedding(vocab, embDim)
	encoder := nn.NewTransformerEncoder(2, embDim, heads, dimFF)
	head := nn.NewLinear(embDim, classes, true)

	params := append([]*tensor.Tensor{}, embedding.Parameters()...)
	params = append(params, encoder.Parameters()...)
	params = append(params, head.Parameters()...)
	opt := optim.NewAdam(params, 5e-3)

	// The post-norm encoder sits on a ~50% plateau for the first couple hundred
	// steps on this parity task before it breaks through to 100%, so train long
	// enough to escape it.
	for step := 0; step < 400; step++ {
		X, Y := makeBatch(32)
		emb := embedding.Forward(X)        // (B, seq, embDim)
		enc := encoder.Forward(emb)        // (B, seq, embDim)
		pooled := enc.MeanAxis(1, false)   // (B, embDim)
		logits := head.Forward(pooled)     // (B, classes)
		loss := nn.CrossEntropyLoss(logits, Y)

		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
		if step%40 == 0 {
			acc := accuracy(logits, Y)
			fmt.Printf("step %3d  loss=%.4f  acc=%.2f%%\n", step, loss.Data[0], acc*100)
		}
	}
}

func makeBatch(batch int) (*tensor.Tensor, *tensor.Tensor) {
	tokens := make([]float64, batch*seqLen)
	targets := make([]float64, batch)
	for b := 0; b < batch; b++ {
		sum := 0
		for s := 0; s < seqLen; s++ {
			tok := rand.Intn(vocab)
			tokens[b*seqLen+s] = float64(tok)
			sum += tok
		}
		targets[b] = float64(sum % 2)
	}
	return tensor.New(tokens, batch, seqLen), tensor.New(targets, batch)
}

func accuracy(logits, targets *tensor.Tensor) float64 {
	preds := logits.ArgMax(1)
	c := 0
	for i, p := range preds.Data {
		if int(p) == int(targets.Data[i]) {
			c++
		}
	}
	return float64(c) / float64(len(targets.Data))
}
