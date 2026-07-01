package nn

import (
	"strconv"

	"gonn/tensor"
)

// TransformerOpt configures transformer layers/stacks. The defaults preserve
// the historical behavior exactly: post-norm, ReLU feed-forward activation,
// no dropout.
type TransformerOpt func(*transformerOpts)

type transformerOpts struct {
	preNorm bool
	dropout float64
	act     *Activation
}

// WithPreNorm selects pre-norm ("norm_first" in PyTorch) residual blocks:
//
//	x = x + attn(LN(x)); x = x + ff(LN(x))
//
// instead of the default post-norm x = LN(x + attn(x)); x = LN(x + ff(x)).
func WithPreNorm() TransformerOpt { return func(o *transformerOpts) { o.preNorm = true } }

// WithTransformerDropout sets dropout probability p at the PyTorch placements:
// after the attention output, after the feed-forward inner activation, and
// after the feed-forward output (each before its residual add). Active only
// in training mode.
func WithTransformerDropout(p float64) TransformerOpt {
	return func(o *transformerOpts) { o.dropout = p }
}

// WithFFActivation sets the feed-forward activation module (default ReLU),
// e.g. WithFFActivation(nn.GELU()).
func WithFFActivation(act *Activation) TransformerOpt {
	return func(o *transformerOpts) { o.act = act }
}

func resolveTransformerOpts(opts []TransformerOpt) transformerOpts {
	var o transformerOpts
	for _, fn := range opts {
		fn(&o)
	}
	if o.act == nil {
		o.act = ReLU()
	}
	return o
}

// TransformerEncoderLayer is one attention + feed-forward block. Default is
// post-norm: x = LN(x + attn(x)); x = LN(x + ff(x)); WithPreNorm() switches
// to x = x + attn(LN(x)); x = x + ff(LN(x)).
type TransformerEncoderLayer struct {
	Base
	Attn    *MultiHeadAttention
	Norm1   *LayerNorm
	FF1     *Linear
	FF2     *Linear
	Norm2   *LayerNorm
	DimFF   int
	EmbDim  int
	PreNorm bool
	Act     *Activation // feed-forward activation (nil = ReLU)
	Drop1   *Dropout    // after attention output
	Drop2   *Dropout    // after feed-forward output
	DropFF  *Dropout    // after feed-forward inner activation
}

// NewTransformerEncoderLayer constructs an encoder block. Options:
// WithPreNorm(), WithTransformerDropout(p), WithFFActivation(act).
func NewTransformerEncoderLayer(embedDim, numHeads, dimFF int, opts ...TransformerOpt) *TransformerEncoderLayer {
	o := resolveTransformerOpts(opts)
	l := &TransformerEncoderLayer{
		Attn:    NewMultiHeadAttention(embedDim, numHeads),
		Norm1:   NewLayerNorm(embedDim),
		FF1:     NewLinear(embedDim, dimFF, true),
		FF2:     NewLinear(dimFF, embedDim, true),
		Norm2:   NewLayerNorm(embedDim),
		DimFF:   dimFF,
		EmbDim:  embedDim,
		PreNorm: o.preNorm,
		Act:     o.act,
		Drop1:   NewDropout(o.dropout),
		Drop2:   NewDropout(o.dropout),
		DropFF:  NewDropout(o.dropout),
	}
	l.regChild("attn", l.Attn)
	l.regChild("norm1", l.Norm1)
	l.regChild("ff1", l.FF1)
	l.regChild("ff2", l.FF2)
	l.regChild("norm2", l.Norm2)
	l.regChild("act", l.Act)
	l.regChild("drop1", l.Drop1)
	l.regChild("drop2", l.Drop2)
	l.regChild("dropff", l.DropFF)
	return l
}

// dropFwd applies d, treating a nil module as identity (zero-value structs).
func dropFwd(d *Dropout, x *tensor.Tensor) *tensor.Tensor {
	if d == nil {
		return x
	}
	return d.Forward(x)
}

// actFwd applies a, defaulting to ReLU when nil (zero-value structs).
func actFwd(a *Activation, x *tensor.Tensor) *tensor.Tensor {
	if a == nil {
		return x.ReLU()
	}
	return a.Forward(x)
}

// ff runs the feed-forward sublayer: FF2(dropout(act(FF1(x)))).
func (l *TransformerEncoderLayer) ff(x *tensor.Tensor) *tensor.Tensor {
	return l.FF2.Forward(dropFwd(l.DropFF, actFwd(l.Act, l.FF1.Forward(x))))
}

// Forward applies attention + FFN with residuals (post-norm by default,
// pre-norm with WithPreNorm()).
func (l *TransformerEncoderLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	if l.PreNorm {
		n := l.Norm1.Forward(x)
		a := l.Attn.Forward(n, n, n, false)
		x = x.Add(dropFwd(l.Drop1, a))
		f := l.ff(l.Norm2.Forward(x))
		return x.Add(dropFwd(l.Drop2, f))
	}
	a := l.Attn.Forward(x, x, x, false)
	x = l.Norm1.Forward(x.Add(dropFwd(l.Drop1, a)))
	f := l.ff(x)
	return l.Norm2.Forward(x.Add(dropFwd(l.Drop2, f)))
}

// TransformerEncoder stacks N encoder layers.
type TransformerEncoder struct {
	Base
	Layers []*TransformerEncoderLayer
}

// NewTransformerEncoder builds N stacked encoder layers. Options are applied
// to every layer.
func NewTransformerEncoder(numLayers, embedDim, numHeads, dimFF int, opts ...TransformerOpt) *TransformerEncoder {
	e := &TransformerEncoder{Layers: make([]*TransformerEncoderLayer, numLayers)}
	for i := range e.Layers {
		e.Layers[i] = NewTransformerEncoderLayer(embedDim, numHeads, dimFF, opts...)
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

// TransformerDecoderLayer is self-attn, cross-attn, FFN (post-norm by
// default; WithPreNorm() selects pre-norm residual blocks).
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
	PreNorm   bool
	Act       *Activation // feed-forward activation (nil = ReLU)
	Drop1     *Dropout    // after self-attention output
	Drop2     *Dropout    // after cross-attention output
	Drop3     *Dropout    // after feed-forward output
	DropFF    *Dropout    // after feed-forward inner activation
}

// NewTransformerDecoderLayer constructs a decoder block. Options:
// WithPreNorm(), WithTransformerDropout(p), WithFFActivation(act).
func NewTransformerDecoderLayer(embedDim, numHeads, dimFF int, opts ...TransformerOpt) *TransformerDecoderLayer {
	o := resolveTransformerOpts(opts)
	l := &TransformerDecoderLayer{
		SelfAttn:  NewMultiHeadAttention(embedDim, numHeads),
		Norm1:     NewLayerNorm(embedDim),
		CrossAttn: NewMultiHeadAttention(embedDim, numHeads),
		Norm2:     NewLayerNorm(embedDim),
		FF1:       NewLinear(embedDim, dimFF, true),
		FF2:       NewLinear(dimFF, embedDim, true),
		Norm3:     NewLayerNorm(embedDim),
		PreNorm:   o.preNorm,
		Act:       o.act,
		Drop1:     NewDropout(o.dropout),
		Drop2:     NewDropout(o.dropout),
		Drop3:     NewDropout(o.dropout),
		DropFF:    NewDropout(o.dropout),
	}
	l.regChild("selfattn", l.SelfAttn)
	l.regChild("norm1", l.Norm1)
	l.regChild("crossattn", l.CrossAttn)
	l.regChild("norm2", l.Norm2)
	l.regChild("ff1", l.FF1)
	l.regChild("ff2", l.FF2)
	l.regChild("norm3", l.Norm3)
	l.regChild("act", l.Act)
	l.regChild("drop1", l.Drop1)
	l.regChild("drop2", l.Drop2)
	l.regChild("drop3", l.Drop3)
	l.regChild("dropff", l.DropFF)
	return l
}

// ff runs the feed-forward sublayer: FF2(dropout(act(FF1(x)))).
func (l *TransformerDecoderLayer) ff(x *tensor.Tensor) *tensor.Tensor {
	return l.FF2.Forward(dropFwd(l.DropFF, actFwd(l.Act, l.FF1.Forward(x))))
}

// Forward runs causal self-attn on tgt, cross-attn over memory, then FFN.
func (l *TransformerDecoderLayer) Forward(tgt, memory *tensor.Tensor) *tensor.Tensor {
	if l.PreNorm {
		n := l.Norm1.Forward(tgt)
		x := tgt.Add(dropFwd(l.Drop1, l.SelfAttn.Forward(n, n, n, true)))
		c := l.CrossAttn.Forward(l.Norm2.Forward(x), memory, memory, false)
		x = x.Add(dropFwd(l.Drop2, c))
		return x.Add(dropFwd(l.Drop3, l.ff(l.Norm3.Forward(x))))
	}
	a := l.SelfAttn.Forward(tgt, tgt, tgt, true)
	x := l.Norm1.Forward(tgt.Add(dropFwd(l.Drop1, a)))
	c := l.CrossAttn.Forward(x, memory, memory, false)
	x = l.Norm2.Forward(x.Add(dropFwd(l.Drop2, c)))
	f := l.ff(x)
	return l.Norm3.Forward(x.Add(dropFwd(l.Drop3, f)))
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

// NewTransformerDecoder builds N stacked decoder layers. Options are applied
// to every layer.
func NewTransformerDecoder(numLayers, embedDim, numHeads, dimFF int, opts ...TransformerOpt) *TransformerDecoder {
	d := &TransformerDecoder{Layers: make([]*TransformerDecoderLayer, numLayers)}
	for i := range d.Layers {
		d.Layers[i] = NewTransformerDecoderLayer(embedDim, numHeads, dimFF, opts...)
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
