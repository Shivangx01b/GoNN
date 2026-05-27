package nn

import (
	"gonn/tensor"
)

// Seq2Seq is a minimal encoder-decoder model:
//
//   - EmbSrc embeds source token indices,
//   - Encoder is a single-layer LSTM that consumes the source sequence; its
//     final (h, c) state initializes the decoder's state,
//   - EmbTgt embeds target token indices,
//   - Decoder is an LSTMCell stepped over the target sequence in teacher-forced
//     fashion,
//   - OutProj is a Linear that maps decoder hidden states to target-vocab logits.
//
// This is intentionally simple: no attention, no beam search, no scheduled
// sampling. v1 returns logits over the whole target sequence in one shot for
// teacher-forced training; for inference, call the encoder once and step the
// decoder cell yourself.
type Seq2Seq struct {
	EmbSrc     *Embedding
	EmbTgt     *Embedding
	Encoder    *LSTMCell // single-layer encoder, run inside Forward
	Decoder    *LSTMCell
	OutProj    *Linear
	HiddenSize int
}

// NewSeq2Seq constructs a Seq2Seq model with single-layer LSTM encoder and
// decoder. embedDim is used for both source and target embeddings.
func NewSeq2Seq(srcVocab, tgtVocab, embedDim, hidden int) *Seq2Seq {
	return &Seq2Seq{
		EmbSrc:     NewEmbedding(srcVocab, embedDim),
		EmbTgt:     NewEmbedding(tgtVocab, embedDim),
		Encoder:    NewLSTMCell(embedDim, hidden),
		Decoder:    NewLSTMCell(embedDim, hidden),
		OutProj:    NewLinear(hidden, tgtVocab, true),
		HiddenSize: hidden,
	}
}

// Forward runs the full teacher-forced pass.
//
// srcIdx: (B, T_src) integer indices stored as float64.
// tgtIdx: (B, T_tgt) integer indices for the decoder inputs (teacher-forced).
//
// Returns logits of shape (B, T_tgt, tgtVocab).
func (m *Seq2Seq) Forward(srcIdx, tgtIdx *tensor.Tensor) *tensor.Tensor {
	if len(srcIdx.Shape) != 2 || len(tgtIdx.Shape) != 2 {
		panic("Seq2Seq.Forward: srcIdx and tgtIdx must be (B, T)")
	}
	B := srcIdx.Shape[0]
	if tgtIdx.Shape[0] != B {
		panic("Seq2Seq.Forward: batch size mismatch between src and tgt")
	}
	Tsrc := srcIdx.Shape[1]
	Ttgt := tgtIdx.Shape[1]

	// --- Encode ---
	srcEmb := m.EmbSrc.Forward(srcIdx) // (B, T_src, E)
	var encState *LSTMState
	for t := 0; t < Tsrc; t++ {
		xt := sliceTime(srcEmb, t) // (B, E)
		encState = m.Encoder.Forward(xt, encState)
	}
	// Initial decoder state = encoder final state.
	decState := encState

	// --- Decode (teacher-forced) ---
	tgtEmb := m.EmbTgt.Forward(tgtIdx) // (B, T_tgt, E)
	H := m.HiddenSize
	outs := make([]*tensor.Tensor, Ttgt)
	for t := 0; t < Ttgt; t++ {
		xt := sliceTime(tgtEmb, t)
		decState = m.Decoder.Forward(xt, decState)
		outs[t] = decState.H
	}
	hidden := stackTime(outs, B, Ttgt, H) // (B, T_tgt, H)

	// --- Project to logits ---
	logits := m.OutProj.Forward(hidden) // (B, T_tgt, tgtVocab)
	return logits
}

// Parameters returns all trainable parameters in the model.
func (m *Seq2Seq) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, m.EmbSrc.Parameters()...)
	ps = append(ps, m.EmbTgt.Parameters()...)
	ps = append(ps, m.Encoder.Parameters()...)
	ps = append(ps, m.Decoder.Parameters()...)
	ps = append(ps, m.OutProj.Parameters()...)
	return ps
}
