package nn

// Eval-mode Conv/Linear + BatchNorm fusion, mirroring
// torch.nn.utils.fusion.{fuse_conv_bn_eval, fuse_linear_bn_eval}.
//
// A BatchNorm in eval mode is an affine map per channel:
//
//	bn(y) = (y - mean) * gamma / sqrt(var + eps) + beta
//
// so it folds into the preceding layer's weight and bias:
//
//	W' = W * gamma / sqrt(var + eps)      (per output channel / row)
//	b' = (b - mean) * gamma / sqrt(var + eps) + beta
//
// The fused layer is a brand-new plain layer (its constructor draws RNG, then
// every element is overwritten), always with a bias — the BN shift needs one
// even when the original layer had none. Fusion is only valid against the
// batch norm's *running* statistics, i.e. it reproduces the eval-mode
// composite; training-mode BN uses batch statistics and cannot be folded.

import (
	"fmt"
	"math"
)

// FuseConvBNEval folds an eval-mode BatchNorm2d into a Conv2d, returning a
// new Conv2d f with f.Forward(x) == bn.Forward(c.Forward(x)) when bn is in
// eval mode. c and bn are not modified.
func FuseConvBNEval(c *Conv2d, bn *BatchNorm2d) *Conv2d {
	if c.OutC != bn.NumFeatures {
		panic(fmt.Sprintf("nn: FuseConvBNEval: conv out channels %d != bn features %d",
			c.OutC, bn.NumFeatures))
	}
	fused := newConv2dLike(c, true)
	block := c.InC * prodInts(c.Kernel) // weight elements per output channel
	for oc := 0; oc < c.OutC; oc++ {
		scale := bn.Weight.Data[oc] / math.Sqrt(bn.RunVar.Data[oc]+bn.Eps)
		for i := 0; i < block; i++ {
			fused.Weight.Data[oc*block+i] = c.Weight.Data[oc*block+i] * scale
		}
		cb := 0.0
		if c.Bias != nil {
			cb = c.Bias.Data[oc]
		}
		fused.Bias.Data[oc] = (cb-bn.RunMean.Data[oc])*scale + bn.Bias.Data[oc]
	}
	return fused
}

// FuseLinearBNEval folds an eval-mode BatchNorm1d into a Linear, returning a
// new Linear f with f.Forward(x) == bn.Forward(l.Forward(x)) when bn is in
// eval mode (for 2D (N, Out) activations). l and bn are not modified.
func FuseLinearBNEval(l *Linear, bn *BatchNorm1d) *Linear {
	if l.OutFeatures != bn.NumFeatures {
		panic(fmt.Sprintf("nn: FuseLinearBNEval: linear out features %d != bn features %d",
			l.OutFeatures, bn.NumFeatures))
	}
	fused := NewLinear(l.InFeatures, l.OutFeatures, true)
	in := l.InFeatures
	for o := 0; o < l.OutFeatures; o++ {
		scale := bn.Weight.Data[o] / math.Sqrt(bn.RunVar.Data[o]+bn.Eps)
		for i := 0; i < in; i++ {
			fused.Weight.Data[o*in+i] = l.Weight.Data[o*in+i] * scale
		}
		lb := 0.0
		if l.Bias != nil {
			lb = l.Bias.Data[o]
		}
		fused.Bias.Data[o] = (lb-bn.RunMean.Data[o])*scale + bn.Bias.Data[o]
	}
	return fused
}
