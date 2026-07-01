package nn

import (
	"gonn/tensor"
)

// Seq2Seq is a minimal encoder-decoder model:
//
//   - EmbSrc embeds source token indices,
//   - Encoder is a single-layer LSTM cell that consumes the source sequence;
//     its final (h, c) state initializes the decoder's state,
//   - EmbTgt embeds target token indices,
//   - Decoder is an LSTMCell stepped over the target sequence in teacher-forced
//     fashion,
//   - OutProj is a Linear that maps decoder hidden states to target-vocab logits.
//
// This is intentionally simple: no attention, no beam search, no scheduled
// sampling. v1 returns logits over the whole target sequence in one shot for
// teacher-forced training; for inference, call the encoder once and step the
// decoder cell yourself.
//
// Seq2Seq takes two inputs, so it satisfies Child but not Module.
type Seq2Seq struct {
	Base
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
	m := &Seq2Seq{
		EmbSrc:     NewEmbedding(srcVocab, embedDim),
		EmbTgt:     NewEmbedding(tgtVocab, embedDim),
		Encoder:    NewLSTMCell(embedDim, hidden),
		Decoder:    NewLSTMCell(embedDim, hidden),
		OutProj:    NewLinear(hidden, tgtVocab, true),
		HiddenSize: hidden,
	}
	m.regChild("embsrc", m.EmbSrc)
	m.regChild("embtgt", m.EmbTgt)
	m.regChild("encoder", m.Encoder)
	m.regChild("decoder", m.Decoder)
	m.regChild("outproj", m.OutProj)
	return m
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
		encState = m.Encoder.Forward(sliceTime(srcEmb, t), encState)
	}
	decState := encState

	// --- Decode (teacher-forced) ---
	tgtEmb := m.EmbTgt.Forward(tgtIdx) // (B, T_tgt, E)
	outs := make([]*tensor.Tensor, Ttgt)
	for t := 0; t < Ttgt; t++ {
		decState = m.Decoder.Forward(sliceTime(tgtEmb, t), decState)
		outs[t] = decState.H
	}
	hidden := stackTime(outs, B, Ttgt, m.HiddenSize) // (B, T_tgt, H)

	// --- Project to logits ---
	return m.OutProj.Forward(hidden) // (B, T_tgt, tgtVocab)
}
