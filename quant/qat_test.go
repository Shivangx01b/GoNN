package quant

import (
	"math"
	"math/rand"
	"testing"

	"gonn/nn"
	"gonn/tensor"
)

// mseLoss builds a differentiable mean-squared-error scalar.
func mseLoss(pred, target *tensor.Tensor) *tensor.Tensor {
	d := pred.Sub(target)
	return d.Mul(d).Sum().MulScalar(1.0 / float64(pred.Numel()))
}

// mseOf computes plain (non-autograd) MSE between two equally shaped tensors.
func mseOf(a, b *tensor.Tensor) float64 {
	s := 0.0
	for i := range a.Data {
		d := a.Data[i] - b.Data[i]
		s += d * d
	}
	return s / float64(len(a.Data))
}

// sgdStep applies vanilla SGD to every parameter and zeroes the grads.
func sgdStep(params []*tensor.Tensor, lr float64) {
	for _, p := range params {
		if p.Grad == nil {
			continue
		}
		for i := range p.Data {
			p.Data[i] -= lr * p.Grad.Data[i]
		}
		p.ZeroGrad()
	}
}

func TestMovingAverageObserverEMA(t *testing.T) {
	// Hand-checked EMA with momentum 0.5.
	obs := MovingAverageMinMaxObserver{Momentum: 0.5}
	obs.Observe(tensor.New([]float64{0, 10}, 2))
	if !obs.Seen() || obs.Min != 0 || obs.Max != 10 {
		t.Fatalf("first observation must initialize directly: min=%g max=%g", obs.Min, obs.Max)
	}
	obs.Observe(tensor.New([]float64{-2, 4}, 2))
	// Min = 0.5*0 + 0.5*(-2) = -1; Max = 0.5*10 + 0.5*4 = 7.
	if obs.Min != -1 || obs.Max != 7 {
		t.Errorf("EMA after second batch: min=%g max=%g, want -1, 7", obs.Min, obs.Max)
	}

	// Zero-value Momentum means the PyTorch default averaging_constant 0.01.
	var def MovingAverageMinMaxObserver
	def.Observe(tensor.New([]float64{0, 1}, 2))
	def.Observe(tensor.New([]float64{0, 2}, 2))
	c := DefaultAveragingConstant
	wantMax := (1-c)*1 + c*2
	if def.Max != wantMax {
		t.Errorf("default-momentum EMA max = %g, want %g", def.Max, wantMax)
	}
	if def.Min != 0 {
		t.Errorf("default-momentum EMA min = %g, want 0", def.Min)
	}

	// ComputeQParams behaves like MinMaxObserver: usable before any
	// observation, affine afterwards with real zero exactly representable.
	var empty MovingAverageMinMaxObserver
	if s, zp := empty.ComputeQParams(); s != 1 || zp != 0 {
		t.Errorf("unobserved qparams = (%g, %d), want (1, 0)", s, zp)
	}
	s, zp := obs.ComputeQParams()
	if s <= 0 || zp < QMin || zp > QMax {
		t.Fatalf("qparams (%g, %d) out of range", s, zp)
	}
	z := (float64(quantizeValue(0, s, zp)) - float64(zp)) * s
	if z != 0 {
		t.Errorf("real 0.0 not exactly representable: %g", z)
	}
}

func TestFakeQuantSTEGradient(t *testing.T) {
	fq := NewFakeQuant(false)
	// Calibrate in training mode on [-1, 1], then freeze.
	fq.Forward(tensor.New([]float64{-1, 1}, 2))
	fq.Eval()

	scale, zp := fq.QParams()
	qminReal := (float64(QMin) - float64(zp)) * scale
	qmaxReal := (float64(QMax) - float64(zp)) * scale
	t.Logf("frozen qparams scale=%g zp=%d, representable range [%g, %g]", scale, zp, qminReal, qmaxReal)

	// Inputs straddling the representable range: three inside, two far out.
	x := tensor.New([]float64{0, 0.5, -0.75, 5, -7}, 5).SetRequiresGrad(true)
	// Non-uniform upstream gradient so "passes grad UNCHANGED" is actually
	// exercised (a plain Sum would only ever propagate 1s).
	up := tensor.New([]float64{2, 3, 4, 5, 6}, 5)
	out := fq.Forward(x)
	out.Mul(up).Sum().Backward()

	if x.Grad == nil {
		t.Fatal("no gradient reached the FakeQuant input")
	}
	wantMask := []float64{1, 1, 1, 0, 0} // fixed expectation for this range
	for i, v := range x.Data {
		want := 0.0
		if v >= qminReal && v <= qmaxReal {
			want = up.Data[i]
		}
		if want != wantMask[i]*up.Data[i] {
			t.Fatalf("test setup drifted: input %g expected mask %g", v, wantMask[i])
		}
		if x.Grad.Data[i] != want {
			t.Errorf("STE grad for x=%g: got %g, want %g (clamped STE)", v, x.Grad.Data[i], want)
		}
	}

	// Forward values must be actual fake-quantization: in-range values within
	// scale/2, out-of-range values clamped to the representable extremes.
	if math.Abs(out.Data[0]-0) > scale/2 || math.Abs(out.Data[1]-0.5) > scale/2 {
		t.Errorf("in-range fake-quant off by more than scale/2: %v", out.Data[:3])
	}
	if out.Data[3] != qmaxReal || out.Data[4] != qminReal {
		t.Errorf("out-of-range inputs must clamp to [%g, %g], got %g, %g",
			qminReal, qmaxReal, out.Data[4], out.Data[3])
	}
}

func TestFakeQuantEvalFreezesQParams(t *testing.T) {
	fq := NewFakeQuant(false)
	fq.Forward(tensor.New([]float64{-1, 1}, 2)) // training-mode calibration
	s0, z0 := fq.QParams()
	mn0, mx0 := fq.Observer.Min, fq.Observer.Max

	fq.Eval()
	shifted := tensor.New([]float64{10, 20, 30}, 3)
	out := fq.Forward(shifted)
	if s1, z1 := fq.QParams(); s1 != s0 || z1 != z0 {
		t.Errorf("eval-mode forward changed qparams: (%g,%d) -> (%g,%d)", s0, z0, s1, z1)
	}
	if fq.Observer.Min != mn0 || fq.Observer.Max != mx0 {
		t.Errorf("eval-mode forward updated the observer: [%g,%g] -> [%g,%g]",
			mn0, mx0, fq.Observer.Min, fq.Observer.Max)
	}
	// Shifted data saturates at the frozen representable max.
	qmaxReal := (float64(QMax) - float64(z0)) * s0
	for i, v := range out.Data {
		if v != qmaxReal {
			t.Errorf("eval output[%d] = %g, want clamp at frozen qmax %g", i, v, qmaxReal)
		}
	}

	// Back in training mode the observer moves again.
	fq.Train()
	fq.Forward(shifted)
	if fq.Observer.Max == mx0 {
		t.Errorf("training-mode forward did not update the observer")
	}
}

func TestFakeQuantSymmetric(t *testing.T) {
	fq := NewFakeQuant(true)
	fq.Forward(tensor.New([]float64{-2, 0.5, 3}, 3))
	s, zp := fq.QParams()
	if zp != 0 {
		t.Errorf("symmetric FakeQuant zero point = %d, want 0", zp)
	}
	want := 3.0 / float64(QMax)
	if math.Abs(s-want) > 1e-15 {
		t.Errorf("symmetric scale = %g, want max|x|/127 = %g", s, want)
	}
}

// qatFixture builds a deterministic regression problem where QAT genuinely
// has something to learn beyond naive rounding:
//   - a weight-range outlier makes the per-tensor weight grid coarse, so
//     weight rounding error dominates;
//   - inputs share a strong common component with a LARGE mean (rank-1
//     dominated second moment), so a row's rounding errors add coherently
//     into a per-row output offset ~ mean(u) * sum(errors). The float bias
//     can absorb that offset — a smooth, continuously trainable gain that
//     QAT finds by gradient descent but naive PTQ (which keeps the
//     float-trained bias) cannot. Coordinated rounding of the sum of errors
//     is a second, discrete gain on the variance term.
//
// It pretrains a float Linear (the standard QAT workflow starts from a
// trained float model) and returns everything the QAT tests need.
type qatFixture struct {
	in, out        int
	xTrain, yTrain *tensor.Tensor
	xTest, yTest   *tensor.Tensor
	float          *nn.Linear // pretrained float model (PTQ baseline)
}

// correlatedInputs samples x[b,k] = u_b + v[b,k] with a shared per-sample
// component u ~ U(1, 3) (mean 2) and small iid v ~ U(-0.5, 0.5).
func correlatedInputs(seed int64, n, in int) *tensor.Tensor {
	rng := rand.New(rand.NewSource(seed))
	x := tensor.Zeros(n, in)
	for b := 0; b < n; b++ {
		u := 1 + 2*rng.Float64()
		for k := 0; k < in; k++ {
			x.Data[b*in+k] = u + (-0.5 + rng.Float64())
		}
	}
	return x
}

func newQATFixture(t *testing.T) *qatFixture {
	t.Helper()
	const in, out, nTrain, nTest = 8, 4, 128, 32
	rng := rand.New(rand.NewSource(21))

	wTrue := make([]float64, out*in)
	for i := range wTrue {
		wTrue[i] = -1.5 + 3*rng.Float64()
	}
	wTrue[0] = 12 // outlier: per-tensor weight scale becomes coarse
	bTrue := make([]float64, out)
	for i := range bTrue {
		bTrue[i] = -0.5 + rng.Float64()
	}
	ref := func(x *tensor.Tensor) *tensor.Tensor {
		n := x.Shape[0]
		y := tensor.Zeros(n, out)
		for b := 0; b < n; b++ {
			for o := 0; o < out; o++ {
				s := bTrue[o]
				for k := 0; k < in; k++ {
					s += x.Data[b*in+k] * wTrue[o*in+k]
				}
				y.Data[b*out+o] = s
			}
		}
		return y
	}
	xTrain := correlatedInputs(22, nTrain, in)
	xTest := correlatedInputs(23, nTest, in)

	f := &qatFixture{
		in: in, out: out,
		xTrain: xTrain, yTrain: ref(xTrain),
		xTest: xTest, yTest: ref(xTest),
	}

	// Pretrain the float model with deterministic init and full-batch SGD.
	// The correlated inputs make the problem ill-conditioned, so this needs
	// plenty of steps to converge along the minor directions.
	f.float = nn.NewLinear(in, out, true)
	initW := seededTensor(24, func(r *rand.Rand) float64 { return -0.3 + 0.6*r.Float64() }, out, in)
	copy(f.float.Weight.Data, initW.Data)
	for i := range f.float.Bias.Data {
		f.float.Bias.Data[i] = 0
	}
	for step := 0; step < 3000; step++ {
		loss := mseLoss(f.float.Forward(f.xTrain), f.yTrain)
		loss.Backward()
		sgdStep(f.float.Parameters(), 0.05)
	}
	if final := mseLoss(f.float.Forward(f.xTrain), f.yTrain).Item(); final > 1e-3 {
		t.Fatalf("float pretraining did not converge: MSE %g", final)
	}
	return f
}

// cloneLinear copies a Linear's parameters into a fresh Linear.
func cloneLinear(l *nn.Linear) *nn.Linear {
	c := nn.NewLinear(l.InFeatures, l.OutFeatures, l.Bias != nil)
	copy(c.Weight.Data, l.Weight.Data)
	if l.Bias != nil {
		copy(c.Bias.Data, l.Bias.Data)
	}
	return c
}

func TestQATLinearTrainConvertAndCompare(t *testing.T) {
	f := newQATFixture(t)

	// --- QAT fine-tune from the pretrained float weights ---
	//
	// Schedule notes (the STE loss landscape is piecewise constant in the
	// weights, so training needs the standard QAT care):
	//   - the bias gets a larger lr than the weights: every discrete rounding
	//     flip of a weight shifts the row output by ~mean(x)*wScale, and the
	//     float bias must re-adapt quickly or the loss limit-cycles;
	//   - after phase 1 the observers are frozen (q.Eval() — the analog of
	//     PyTorch's disable_observer near the end of QAT; the FakeQuants keep
	//     fake-quantizing and passing STE gradients) and only the bias keeps
	//     polishing, which is smooth and convex, so no late rounding flip can
	//     leave the bias stranded mid-adaptation.
	q := NewQATLinearFrom(cloneLinear(f.float))
	wBefore := append([]float64(nil), q.L.Weight.Data...)

	const (
		phase1, phase2 = 500, 150
		lrW, lrB       = 0.005, 0.05
	)
	firstLoss, lastLoss := math.NaN(), math.NaN()
	for step := 0; step < phase1+phase2; step++ {
		if step == phase1 {
			q.Eval() // freeze observers/qparams for the polish phase
		}
		loss := mseLoss(q.Forward(f.xTrain), f.yTrain)
		if step == 0 {
			firstLoss = loss.Item()
		}
		lastLoss = loss.Item()
		loss.Backward()
		if step == 0 {
			// The STE must deliver a real gradient into the float weight.
			if q.L.Weight.Grad == nil || maxAbs(q.L.Weight.Grad.Data) == 0 {
				t.Fatal("no gradient reached the float weight through FakeQuant")
			}
		}
		if step < phase1 {
			sgdStep([]*tensor.Tensor{q.L.Weight}, lrW)
		} else {
			q.L.Weight.ZeroGrad()
		}
		sgdStep([]*tensor.Tensor{q.L.Bias}, lrB)
	}
	t.Logf("QAT fine-tune loss: first=%g last=%g", firstLoss, lastLoss)
	if !(lastLoss < firstLoss) {
		t.Errorf("QAT training loss did not decrease: first=%g last=%g", firstLoss, lastLoss)
	}
	moved := 0.0
	for i := range wBefore {
		if d := math.Abs(q.L.Weight.Data[i] - wBefore[i]); d > moved {
			moved = d
		}
	}
	if moved == 0 {
		t.Error("QAT training never updated the float weight")
	}

	// --- Convert and check the frozen model matches eval-mode QAT ---
	q.Eval()
	evalOut := q.Forward(f.xTest)
	conv := q.Convert()
	convOut := conv.Forward(f.xTest)

	inScale, _ := q.InputFQ.QParams()
	wScale, wzp := q.WeightFQ.QParams()
	if wzp != 0 {
		t.Errorf("weight FakeQuant zero point = %d, want 0 (symmetric)", wzp)
	}
	maxDiff := 0.0
	for i := range evalOut.Data {
		if d := math.Abs(evalOut.Data[i] - convOut.Data[i]); d > maxDiff {
			maxDiff = d
		}
	}
	bound := 2 * inScale * wScale
	t.Logf("convert vs eval: max abs diff = %g (bound 2*sx*sw = %g)", maxDiff, bound)
	if maxDiff > bound {
		t.Errorf("converted StaticLinear diverges from QAT eval output: max abs diff %g > %g", maxDiff, bound)
	}
	// Both paths compute identical quantized integers, so the residual is
	// pure float64 summation-order noise — orders of magnitude below the
	// quantization step.
	if maxDiff > 1e-8 {
		t.Errorf("converted output should match eval output to float rounding, max abs diff %g", maxDiff)
	}
	if conv.InScale != inScale || conv.WScale != wScale {
		t.Errorf("Convert must freeze the observers' qparams: got (sx=%g, sw=%g), want (%g, %g)",
			conv.InScale, conv.WScale, inScale, wScale)
	}

	// --- QAT beats naive post-training quantization of the float twin ---
	var obs MinMaxObserver
	obs.Observe(f.xTrain) // calibrate PTQ on the same training inputs
	ptq := NewStaticLinearFrom(f.float, &obs)

	qatMSE := mseOf(convOut, f.yTest)
	ptqMSE := mseOf(ptq.Forward(f.xTest), f.yTest)
	floatMSE := mseOf(f.float.Forward(f.xTest), f.yTest)
	t.Logf("test MSE vs ground truth: float=%g QAT-converted=%g naive-PTQ=%g", floatMSE, qatMSE, ptqMSE)
	if !(qatMSE < ptqMSE) {
		t.Errorf("QAT-converted MSE %g not lower than naive PTQ MSE %g", qatMSE, ptqMSE)
	}

	// --- Per-channel convert option: sanity ---
	// Note: no accuracy assertion vs the per-tensor convert. QAT trained the
	// weights against the per-tensor grid, so re-quantizing them on a
	// different (per-channel) grid discards that adaptation; per-channel
	// converts of per-tensor-trained QAT models are not expected to win.
	convPC := q.Convert(WithPerChannelWeights())
	if !convPC.PerChannel {
		t.Fatal("Convert(WithPerChannelWeights()) must build a per-channel core")
	}
	pcOut := convPC.Forward(f.xTest)
	if len(pcOut.Data) != len(evalOut.Data) {
		t.Fatalf("per-channel converted output shape %v", pcOut.Shape)
	}
	t.Logf("per-channel converted test MSE = %g", mseOf(pcOut, f.yTest))
}

func TestQATLinearModeAndParamTree(t *testing.T) {
	q := NewQATLinearFrom(nn.NewLinear(4, 3, true))
	// Weight and bias must be reachable through the QATLinear parameter tree.
	if got := len(q.Parameters()); got != 2 {
		t.Fatalf("QATLinear.Parameters() has %d tensors, want 2 (weight, bias)", got)
	}
	// Train/eval mode must propagate into both FakeQuants.
	q.Eval()
	if q.WeightFQ.Training() || q.InputFQ.Training() {
		t.Error("Eval() did not propagate to the FakeQuant children")
	}
	q.Train()
	if !q.WeightFQ.Training() || !q.InputFQ.Training() {
		t.Error("Train() did not propagate to the FakeQuant children")
	}
}
