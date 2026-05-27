package nn

import "gonn/tensor"

// TransformerEncoderLayer uses post-norm: x = LN(x + attn(x)); x = LN(x + ff(x)).
type TransformerEncoderLayer struct {
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
	return &TransformerEncoderLayer{
		Attn:   NewMultiHeadAttention(embedDim, numHeads),
		Norm1:  NewLayerNorm(embedDim),
		FF1:    NewLinear(embedDim, dimFF, true),
		FF2:    NewLinear(dimFF, embedDim, true),
		Norm2:  NewLayerNorm(embedDim),
		DimFF:  dimFF,
		EmbDim: embedDim,
	}
}

// Forward applies attention + FFN with residuals and post-norm.
func (l *TransformerEncoderLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	a := l.Attn.Forward(x, x, x, false)
	x = l.Norm1.Forward(x.Add(a))
	f := l.FF2.Forward(l.FF1.Forward(x).ReLU())
	return l.Norm2.Forward(x.Add(f))
}

// Parameters returns all sub-module parameters.
func (l *TransformerEncoderLayer) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, l.Attn.Parameters()...)
	ps = append(ps, l.Norm1.Parameters()...)
	ps = append(ps, l.FF1.Parameters()...)
	ps = append(ps, l.FF2.Parameters()...)
	ps = append(ps, l.Norm2.Parameters()...)
	return ps
}

// TransformerEncoder stacks N encoder layers.
type TransformerEncoder struct {
	Layers []*TransformerEncoderLayer
}

// NewTransformerEncoder builds N stacked encoder layers.
func NewTransformerEncoder(numLayers, embedDim, numHeads, dimFF int) *TransformerEncoder {
	ls := make([]*TransformerEncoderLayer, numLayers)
	for i := range ls {
		ls[i] = NewTransformerEncoderLayer(embedDim, numHeads, dimFF)
	}
	return &TransformerEncoder{Layers: ls}
}

// Forward applies each encoder layer in sequence.
func (e *TransformerEncoder) Forward(x *tensor.Tensor) *tensor.Tensor {
	for _, l := range e.Layers {
		x = l.Forward(x)
	}
	return x
}

// Parameters returns all layer parameters.
func (e *TransformerEncoder) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range e.Layers {
		ps = append(ps, l.Parameters()...)
	}
	return ps
}

// TransformerDecoderLayer is post-norm: self-attn, cross-attn, FFN.
type TransformerDecoderLayer struct {
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
	return &TransformerDecoderLayer{
		SelfAttn:  NewMultiHeadAttention(embedDim, numHeads),
		Norm1:     NewLayerNorm(embedDim),
		CrossAttn: NewMultiHeadAttention(embedDim, numHeads),
		Norm2:     NewLayerNorm(embedDim),
		FF1:       NewLinear(embedDim, dimFF, true),
		FF2:       NewLinear(dimFF, embedDim, true),
		Norm3:     NewLayerNorm(embedDim),
	}
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

// Parameters returns all sub-module parameters.
func (l *TransformerDecoderLayer) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, l.SelfAttn.Parameters()...)
	ps = append(ps, l.Norm1.Parameters()...)
	ps = append(ps, l.CrossAttn.Parameters()...)
	ps = append(ps, l.Norm2.Parameters()...)
	ps = append(ps, l.FF1.Parameters()...)
	ps = append(ps, l.FF2.Parameters()...)
	ps = append(ps, l.Norm3.Parameters()...)
	return ps
}

// Forward — Module interface — uses tgt as both inputs (self-encoding). Use
// the two-arg Forward for actual encoder-decoder use.
func (l *TransformerDecoderLayer) ForwardModule(x *tensor.Tensor) *tensor.Tensor {
	return l.Forward(x, x)
}

// TransformerDecoder stacks N decoder layers over a shared memory.
type TransformerDecoder struct {
	Layers []*TransformerDecoderLayer
}

// NewTransformerDecoder builds N stacked decoder layers.
func NewTransformerDecoder(numLayers, embedDim, numHeads, dimFF int) *TransformerDecoder {
	ls := make([]*TransformerDecoderLayer, numLayers)
	for i := range ls {
		ls[i] = NewTransformerDecoderLayer(embedDim, numHeads, dimFF)
	}
	return &TransformerDecoder{Layers: ls}
}

// Forward applies each decoder layer over (tgt, memory).
func (d *TransformerDecoder) Forward(tgt, memory *tensor.Tensor) *tensor.Tensor {
	for _, l := range d.Layers {
		tgt = l.Forward(tgt, memory)
	}
	return tgt
}

// Parameters returns all layer parameters.
func (d *TransformerDecoder) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range d.Layers {
		ps = append(ps, l.Parameters()...)
	}
	return ps
}
