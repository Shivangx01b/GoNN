package nn

import (
	"strconv"

	"gonn/tensor"
)

// TransformerEncoderLayer uses post-norm: x = LN(x + attn(x)); x = LN(x + ff(x)).
type TransformerEncoderLayer struct {
	Base
	Attn   *MultiHeadAttention
	Norm1  *LayerNorm
	FF1    *Linear
	FF2    *Linear
	Norm2  *LayerNorm
	DimFF  int
	EmbDim int
}

// NewTransformerEncoderLayer constructs an encoder block.
func NewTransformerEncoderLayer(embedDim, numHeads, dimFF int) *TransformerEncoderLayer {
	l := &TransformerEncoderLayer{
		Attn:   NewMultiHeadAttention(embedDim, numHeads),
		Norm1:  NewLayerNorm(embedDim),
		FF1:    NewLinear(embedDim, dimFF, true),
		FF2:    NewLinear(dimFF, embedDim, true),
		Norm2:  NewLayerNorm(embedDim),
		DimFF:  dimFF,
		EmbDim: embedDim,
	}
	l.regChild("attn", l.Attn)
	l.regChild("norm1", l.Norm1)
	l.regChild("ff1", l.FF1)
	l.regChild("ff2", l.FF2)
	l.regChild("norm2", l.Norm2)
	return l
}

// Forward applies attention + FFN with residuals and post-norm.
func (l *TransformerEncoderLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	a := l.Attn.Forward(x, x, x, false)
	x = l.Norm1.Forward(x.Add(a))
	f := l.FF2.Forward(l.FF1.Forward(x).ReLU())
	return l.Norm2.Forward(x.Add(f))
}

// TransformerEncoder stacks N encoder layers.
type TransformerEncoder struct {
	Base
	Layers []*TransformerEncoderLayer
}

// NewTransformerEncoder builds N stacked encoder layers.
func NewTransformerEncoder(numLayers, embedDim, numHeads, dimFF int) *TransformerEncoder {
	e := &TransformerEncoder{Layers: make([]*TransformerEncoderLayer, numLayers)}
	for i := range e.Layers {
		e.Layers[i] = NewTransformerEncoderLayer(embedDim, numHeads, dimFF)
		e.regChild("layers."+strconv.Itoa(i), e.Layers[i])
	}
	return e
}

// Forward applies each encoder layer in sequence.
func (e *TransformerEncoder) Forward(x *tensor.Tensor) *tensor.Tensor {
	for _, l := range e.Layers {
		x = l.Forward(x)
	}
	return x
}

// TransformerDecoderLayer is post-norm: self-attn, cross-attn, FFN.
// It takes (tgt, memory), so it satisfies Child but not Module; ForwardModule
// adapts it to single-input use.
type TransformerDecoderLayer struct {
	Base
	SelfAttn  *MultiHeadAttention
	Norm1     *LayerNorm
	CrossAttn *MultiHeadAttention
	Norm2     *LayerNorm
	FF1       *Linear
	FF2       *Linear
	Norm3     *LayerNorm
}

// NewTransformerDecoderLayer constructs a decoder block.
func NewTransformerDecoderLayer(embedDim, numHeads, dimFF int) *TransformerDecoderLayer {
	l := &TransformerDecoderLayer{
		SelfAttn:  NewMultiHeadAttention(embedDim, numHeads),
		Norm1:     NewLayerNorm(embedDim),
		CrossAttn: NewMultiHeadAttention(embedDim, numHeads),
		Norm2:     NewLayerNorm(embedDim),
		FF1:       NewLinear(embedDim, dimFF, true),
		FF2:       NewLinear(dimFF, embedDim, true),
		Norm3:     NewLayerNorm(embedDim),
	}
	l.regChild("selfattn", l.SelfAttn)
	l.regChild("norm1", l.Norm1)
	l.regChild("crossattn", l.CrossAttn)
	l.regChild("norm2", l.Norm2)
	l.regChild("ff1", l.FF1)
	l.regChild("ff2", l.FF2)
	l.regChild("norm3", l.Norm3)
	return l
}

// Forward runs causal self-attn on tgt, cross-attn over memory, then FFN.
func (l *TransformerDecoderLayer) Forward(tgt, memory *tensor.Tensor) *tensor.Tensor {
	a := l.SelfAttn.Forward(tgt, tgt, tgt, true)
	x := l.Norm1.Forward(tgt.Add(a))
	c := l.CrossAttn.Forward(x, memory, memory, false)
	x = l.Norm2.Forward(x.Add(c))
	f := l.FF2.Forward(l.FF1.Forward(x).ReLU())
	return l.Norm3.Forward(x.Add(f))
}

// ForwardModule — Module-style single input — uses tgt as both inputs
// (self-encoding). Use the two-arg Forward for actual encoder-decoder use.
func (l *TransformerDecoderLayer) ForwardModule(x *tensor.Tensor) *tensor.Tensor {
	return l.Forward(x, x)
}

// TransformerDecoder stacks N decoder layers over a shared memory.
type TransformerDecoder struct {
	Base
	Layers []*TransformerDecoderLayer
}

// NewTransformerDecoder builds N stacked decoder layers.
func NewTransformerDecoder(numLayers, embedDim, numHeads, dimFF int) *TransformerDecoder {
	d := &TransformerDecoder{Layers: make([]*TransformerDecoderLayer, numLayers)}
	for i := range d.Layers {
		d.Layers[i] = NewTransformerDecoderLayer(embedDim, numHeads, dimFF)
		d.regChild("layers."+strconv.Itoa(i), d.Layers[i])
	}
	return d
}

// Forward applies each decoder layer over (tgt, memory).
func (d *TransformerDecoder) Forward(tgt, memory *tensor.Tensor) *tensor.Tensor {
	for _, l := range d.Layers {
		tgt = l.Forward(tgt, memory)
	}
	return tgt
}
