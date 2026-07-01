package nn

import "gonn/tensor"

// Transformer is the all-in-one encoder-decoder transformer, mirroring
// torch.nn.Transformer's structure (batch-first): a TransformerEncoder stack,
// a TransformerDecoder stack, and — like PyTorch — a final LayerNorm after
// each stack (PyTorch always constructs encoder_norm/decoder_norm).
//
// The decoder's self-attention is always causal, equivalent to PyTorch's
// standard generate_square_subsequent_mask tgt_mask. Transformer takes two
// inputs, so it satisfies Child but not Module.
type Transformer struct {
	Base
	Encoder  *TransformerEncoder
	Decoder  *TransformerDecoder
	EncNorm  *LayerNorm
	DecNorm  *LayerNorm
	EmbedDim int
	NumHeads int
}

// NewTransformer builds the full model. Options (WithPreNorm,
// WithTransformerDropout, WithFFActivation) are applied to every encoder and
// decoder layer.
func NewTransformer(embedDim, numHeads, numEncoderLayers, numDecoderLayers, dimFF int, opts ...TransformerOpt) *Transformer {
	t := &Transformer{
		Encoder:  NewTransformerEncoder(numEncoderLayers, embedDim, numHeads, dimFF, opts...),
		Decoder:  NewTransformerDecoder(numDecoderLayers, embedDim, numHeads, dimFF, opts...),
		EncNorm:  NewLayerNorm(embedDim),
		DecNorm:  NewLayerNorm(embedDim),
		EmbedDim: embedDim,
		NumHeads: numHeads,
	}
	t.regChild("encoder", t.Encoder)
	t.regChild("decoder", t.Decoder)
	t.regChild("encnorm", t.EncNorm)
	t.regChild("decnorm", t.DecNorm)
	return t
}

// Forward runs the encoder over src, then the decoder over (tgt, memory).
// src: (B, Tsrc, E); tgt: (B, Ttgt, E). Returns (B, Ttgt, E).
func (t *Transformer) Forward(src, tgt *tensor.Tensor) *tensor.Tensor {
	memory := t.EncNorm.Forward(t.Encoder.Forward(src))
	return t.DecNorm.Forward(t.Decoder.Forward(tgt, memory))
}
