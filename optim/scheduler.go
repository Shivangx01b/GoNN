package optim

import "math"

// Scheduler adjusts an Optimizer's learning rate on every call to Step.
type Scheduler interface {
	Step()
}

// MetricScheduler is implemented by schedulers driven by an observed metric
// (ReduceLROnPlateau). It intentionally does not satisfy Scheduler — the
// caller must supply the metric — matching PyTorch's split.
type MetricScheduler interface {
	Step(metric float64)
}

// schedulerBase gives schedulers group-aware LR control. Two application
// modes, chosen per scheduler to preserve the historical semantics exactly:
//
//   - scaleLRs (recursive): multiply every group's CURRENT lr — used by
//     Step/MultiStep/Exponential/Plateau, whose decays deliberately compound
//     (also with other schedulers, e.g. via ChainedScheduler).
//   - setLRs (closed-form): set each group's lr to a function of its BASE lr
//     captured at construction — used by Cosine/Linear/Polynomial/OneCycle,
//     which are idempotent in the step count.
//
// With one group both modes degenerate to the historical SetLR calls; with
// several groups the per-group LR ratios survive scheduling.
type schedulerBase struct {
	opt     Optimizer
	baseLRs []float64 // one per group, captured at construction
}

func newSchedulerBase(opt Optimizer) schedulerBase {
	groups := opt.Groups()
	base := make([]float64, len(groups))
	for i, g := range groups {
		base[i] = g.LR
	}
	return schedulerBase{opt: opt, baseLRs: base}
}

// scaleLRs multiplies every group's current lr by gamma.
func (s *schedulerBase) scaleLRs(gamma float64) {
	for _, g := range s.opt.Groups() {
		g.LR *= gamma
	}
}

// setLRs sets each group's lr to f(baseLR_i).
func (s *schedulerBase) setLRs(f func(baseLR float64) float64) {
	for i, g := range s.opt.Groups() {
		g.LR = f(s.baseLRs[i])
	}
}

// StepLR decays the learning rate by gamma every stepSize calls.
type StepLR struct {
	schedulerBase
	stepSize int
	gamma    float64
	count    int
}

// NewStepLR constructs a StepLR scheduler.
func NewStepLR(opt Optimizer, stepSize int, gamma float64) *StepLR {
	return &StepLR{schedulerBase: newSchedulerBase(opt), stepSize: stepSize, gamma: gamma}
}

// Step advances the scheduler one tick.
func (s *StepLR) Step() {
	s.count++
	if s.stepSize > 0 && s.count%s.stepSize == 0 {
		s.scaleLRs(s.gamma)
	}
}

// MultiStepLR decays the learning rate by gamma at each milestone.
type MultiStepLR struct {
	schedulerBase
	milestones []int
	gamma      float64
	count      int
}

// NewMultiStepLR constructs a MultiStepLR scheduler.
func NewMultiStepLR(opt Optimizer, milestones []int, gamma float64) *MultiStepLR {
	return &MultiStepLR{
		schedulerBase: newSchedulerBase(opt),
		milestones:    append([]int(nil), milestones...),
		gamma:         gamma,
	}
}

// Step advances the scheduler one tick.
func (s *MultiStepLR) Step() {
	s.count++
	for _, m := range s.milestones {
		if m == s.count {
			s.scaleLRs(s.gamma)
			return
		}
	}
}

// ExponentialLR multiplies the learning rate by gamma every step.
type ExponentialLR struct {
	schedulerBase
	gamma float64
}

// NewExponentialLR constructs an ExponentialLR scheduler.
func NewExponentialLR(opt Optimizer, gamma float64) *ExponentialLR {
	return &ExponentialLR{schedulerBase: newSchedulerBase(opt), gamma: gamma}
}

// Step advances the scheduler one tick.
func (s *ExponentialLR) Step() { s.scaleLRs(s.gamma) }

// CosineAnnealingLR follows a half-cosine from each group's initial lr down
// to etaMin over TMax steps and then continues to oscillate.
type CosineAnnealingLR struct {
	schedulerBase
	TMax   int
	etaMin float64
	tCur   int
}

// NewCosineAnnealingLR constructs a CosineAnnealingLR scheduler.
func NewCosineAnnealingLR(opt Optimizer, TMax int, etaMin float64) *CosineAnnealingLR {
	return &CosineAnnealingLR{schedulerBase: newSchedulerBase(opt), TMax: TMax, etaMin: etaMin}
}

// Step advances the scheduler one tick.
func (s *CosineAnnealingLR) Step() {
	s.tCur++
	if s.TMax <= 0 {
		return
	}
	cos := math.Cos(math.Pi * float64(s.tCur) / float64(s.TMax))
	s.setLRs(func(base float64) float64 {
		return s.etaMin + 0.5*(base-s.etaMin)*(1+cos)
	})
}

// LinearLR linearly scales each group's learning rate from startFactor*baseLR
// to endFactor*baseLR over totalIters steps.
type LinearLR struct {
	schedulerBase
	startFactor float64
	endFactor   float64
	totalIters  int
	count       int
}

// NewLinearLR constructs a LinearLR scheduler.
func NewLinearLR(opt Optimizer, startFactor, endFactor float64, totalIters int) *LinearLR {
	s := &LinearLR{
		schedulerBase: newSchedulerBase(opt),
		startFactor:   startFactor,
		endFactor:     endFactor,
		totalIters:    totalIters,
	}
	// Apply start factor immediately.
	s.setLRs(func(base float64) float64 { return base * startFactor })
	return s
}

// Step advances the scheduler one tick.
func (s *LinearLR) Step() {
	s.count++
	factor := s.endFactor
	if s.totalIters > 0 && s.count < s.totalIters {
		frac := float64(s.count) / float64(s.totalIters)
		factor = s.startFactor + (s.endFactor-s.startFactor)*frac
	}
	s.setLRs(func(base float64) float64 { return base * factor })
}

// ReduceLROnPlateau reduces the learning rate when a metric has stopped
// improving. It is a MetricScheduler: call Step(lossValue) with the observed
// metric.
type ReduceLROnPlateau struct {
	schedulerBase
	factor    float64
	threshold float64
	patience  int
	bestLoss  float64
	numBad    int
	hasBest   bool
}

// NewReduceLROnPlateau constructs a ReduceLROnPlateau scheduler.
func NewReduceLROnPlateau(opt Optimizer, factor, threshold float64, patience int) *ReduceLROnPlateau {
	return &ReduceLROnPlateau{
		schedulerBase: newSchedulerBase(opt),
		factor:        factor,
		threshold:     threshold,
		patience:      patience,
	}
}

// Step records the latest loss value and reduces the learning rate if no
// improvement has been seen within `patience` calls.
func (s *ReduceLROnPlateau) Step(lossValue float64) {
	if !s.hasBest {
		s.bestLoss = lossValue
		s.hasBest = true
		s.numBad = 0
		return
	}
	if lossValue < s.bestLoss-s.threshold {
		s.bestLoss = lossValue
		s.numBad = 0
		return
	}
	s.numBad++
	if s.numBad > s.patience {
		s.scaleLRs(s.factor)
		s.numBad = 0
	}
}

// PolynomialLR decays each group's lr as baseLR * (1 - step/totalIters)^power.
// Once `count` reaches totalIters the lr is clamped at 0.
type PolynomialLR struct {
	schedulerBase
	totalIters int
	power      float64
	count      int
}

// NewPolynomialLR constructs a PolynomialLR scheduler.
func NewPolynomialLR(opt Optimizer, totalIters int, power float64) *PolynomialLR {
	return &PolynomialLR{schedulerBase: newSchedulerBase(opt), totalIters: totalIters, power: power}
}

// Step advances the scheduler one tick.
func (s *PolynomialLR) Step() {
	s.count++
	if s.totalIters <= 0 {
		return
	}
	if s.count >= s.totalIters {
		s.setLRs(func(float64) float64 { return 0 })
		return
	}
	frac := 1 - float64(s.count)/float64(s.totalIters)
	scale := math.Pow(frac, s.power)
	s.setLRs(func(base float64) float64 { return base * scale })
}

// ChainedScheduler steps a list of schedulers in order on each call.
type ChainedScheduler struct {
	schedulers []Scheduler
}

// NewChainedScheduler constructs a ChainedScheduler.
func NewChainedScheduler(schedulers ...Scheduler) *ChainedScheduler {
	return &ChainedScheduler{schedulers: append([]Scheduler(nil), schedulers...)}
}

// Step advances every wrapped scheduler one tick, in order.
func (s *ChainedScheduler) Step() {
	for _, sub := range s.schedulers {
		sub.Step()
	}
}

// SequentialLR runs schedulers[k] until milestones[k] is reached, then
// switches to schedulers[k+1]. len(milestones) must equal len(schedulers)-1.
type SequentialLR struct {
	schedulers []Scheduler
	milestones []int
	count      int
}

// NewSequentialLR constructs a SequentialLR scheduler.
func NewSequentialLR(schedulers []Scheduler, milestones []int) *SequentialLR {
	return &SequentialLR{
		schedulers: append([]Scheduler(nil), schedulers...),
		milestones: append([]int(nil), milestones...),
	}
}

// Step advances the active sub-scheduler. The active index advances past each
// milestone as the running counter crosses it.
func (s *SequentialLR) Step() {
	s.count++
	idx := 0
	for i, m := range s.milestones {
		if s.count > m {
			idx = i + 1
		} else {
			break
		}
	}
	if idx >= len(s.schedulers) {
		idx = len(s.schedulers) - 1
	}
	if idx < 0 || idx >= len(s.schedulers) {
		return
	}
	s.schedulers[idx].Step()
}

// CyclicLR follows a triangular schedule between baseLR and maxLR (scalar
// bounds broadcast to every group, matching PyTorch's scalar form). Each half
// cycle lasts stepSize calls: the lr ramps up for stepSize steps, then down
// for stepSize steps, then repeats.
type CyclicLR struct {
	schedulerBase
	baseLR   float64
	maxLR    float64
	stepSize int
	count    int
}

// NewCyclicLR constructs a CyclicLR scheduler.
func NewCyclicLR(opt Optimizer, baseLR, maxLR float64, stepSize int) *CyclicLR {
	c := &CyclicLR{
		schedulerBase: newSchedulerBase(opt),
		baseLR:        baseLR,
		maxLR:         maxLR,
		stepSize:      stepSize,
	}
	c.setLRs(func(float64) float64 { return baseLR })
	return c
}

// Step advances the scheduler one tick.
func (s *CyclicLR) Step() {
	s.count++
	if s.stepSize <= 0 {
		return
	}
	cycle := math.Floor(1 + float64(s.count)/(2*float64(s.stepSize)))
	x := math.Abs(float64(s.count)/float64(s.stepSize) - 2*cycle + 1)
	lr := s.baseLR + (s.maxLR-s.baseLR)*math.Max(0, 1-x)
	s.setLRs(func(float64) float64 { return lr })
}

// OneCycleLR is a simple triangular one-cycle schedule that ramps each
// group's lr from its initial value up to the shared maxLR at the midpoint of
// totalSteps and back down to the initial value.
type OneCycleLR struct {
	schedulerBase
	maxLR      float64
	totalSteps int
	count      int
}

// NewOneCycleLR constructs a OneCycleLR scheduler.
func NewOneCycleLR(opt Optimizer, maxLR float64, totalSteps int) *OneCycleLR {
	return &OneCycleLR{schedulerBase: newSchedulerBase(opt), maxLR: maxLR, totalSteps: totalSteps}
}

// Step advances the scheduler one tick.
func (s *OneCycleLR) Step() {
	s.count++
	if s.totalSteps <= 0 {
		return
	}
	half := float64(s.totalSteps) / 2
	t := float64(s.count)
	s.setLRs(func(base float64) float64 {
		switch {
		case t <= half:
			// Ramp up: baseLR -> maxLR
			frac := t / half
			return base + (s.maxLR-base)*frac
		case t <= float64(s.totalSteps):
			// Ramp down: maxLR -> baseLR
			frac := (t - half) / half
			return s.maxLR + (base-s.maxLR)*frac
		default:
			return base
		}
	})
}
