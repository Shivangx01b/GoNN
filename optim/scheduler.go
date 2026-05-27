package optim

import "math"

// Scheduler adjusts an Optimizer's learning rate on every call to Step.
type Scheduler interface {
	Step()
}

// StepLR decays the learning rate by gamma every stepSize calls.
type StepLR struct {
	opt      Optimizer
	stepSize int
	gamma    float64
	count    int
}

// NewStepLR constructs a StepLR scheduler.
func NewStepLR(opt Optimizer, stepSize int, gamma float64) *StepLR {
	return &StepLR{opt: opt, stepSize: stepSize, gamma: gamma}
}

// Step advances the scheduler one tick.
func (s *StepLR) Step() {
	s.count++
	if s.stepSize > 0 && s.count%s.stepSize == 0 {
		s.opt.SetLR(s.opt.LR() * s.gamma)
	}
}

// MultiStepLR decays the learning rate by gamma at each milestone.
type MultiStepLR struct {
	opt        Optimizer
	milestones []int
	gamma      float64
	count      int
}

// NewMultiStepLR constructs a MultiStepLR scheduler.
func NewMultiStepLR(opt Optimizer, milestones []int, gamma float64) *MultiStepLR {
	ms := append([]int(nil), milestones...)
	return &MultiStepLR{opt: opt, milestones: ms, gamma: gamma}
}

// Step advances the scheduler one tick.
func (s *MultiStepLR) Step() {
	s.count++
	for _, m := range s.milestones {
		if m == s.count {
			s.opt.SetLR(s.opt.LR() * s.gamma)
			return
		}
	}
}

// ExponentialLR multiplies the learning rate by gamma every step.
type ExponentialLR struct {
	opt   Optimizer
	gamma float64
}

// NewExponentialLR constructs an ExponentialLR scheduler.
func NewExponentialLR(opt Optimizer, gamma float64) *ExponentialLR {
	return &ExponentialLR{opt: opt, gamma: gamma}
}

// Step advances the scheduler one tick.
func (s *ExponentialLR) Step() { s.opt.SetLR(s.opt.LR() * s.gamma) }

// CosineAnnealingLR follows a half-cosine from the initial lr down to etaMin
// over TMax steps and then continues to oscillate.
type CosineAnnealingLR struct {
	opt     Optimizer
	TMax    int
	etaMin  float64
	baseLR  float64
	tCur    int
}

// NewCosineAnnealingLR constructs a CosineAnnealingLR scheduler.
func NewCosineAnnealingLR(opt Optimizer, TMax int, etaMin float64) *CosineAnnealingLR {
	return &CosineAnnealingLR{opt: opt, TMax: TMax, etaMin: etaMin, baseLR: opt.LR()}
}

// Step advances the scheduler one tick.
func (s *CosineAnnealingLR) Step() {
	s.tCur++
	if s.TMax <= 0 {
		return
	}
	cos := math.Cos(math.Pi * float64(s.tCur) / float64(s.TMax))
	lr := s.etaMin + 0.5*(s.baseLR-s.etaMin)*(1+cos)
	s.opt.SetLR(lr)
}

// LinearLR linearly scales the learning rate from startFactor*baseLR to
// endFactor*baseLR over totalIters steps.
type LinearLR struct {
	opt         Optimizer
	startFactor float64
	endFactor   float64
	totalIters  int
	baseLR      float64
	count       int
}

// NewLinearLR constructs a LinearLR scheduler.
func NewLinearLR(opt Optimizer, startFactor, endFactor float64, totalIters int) *LinearLR {
	s := &LinearLR{
		opt:         opt,
		startFactor: startFactor,
		endFactor:   endFactor,
		totalIters:  totalIters,
		baseLR:      opt.LR(),
	}
	// Apply start factor immediately.
	s.opt.SetLR(s.baseLR * s.startFactor)
	return s
}

// Step advances the scheduler one tick.
func (s *LinearLR) Step() {
	s.count++
	if s.totalIters <= 0 {
		s.opt.SetLR(s.baseLR * s.endFactor)
		return
	}
	if s.count >= s.totalIters {
		s.opt.SetLR(s.baseLR * s.endFactor)
		return
	}
	frac := float64(s.count) / float64(s.totalIters)
	factor := s.startFactor + (s.endFactor-s.startFactor)*frac
	s.opt.SetLR(s.baseLR * factor)
}

// ReduceLROnPlateau reduces the learning rate when a metric has stopped improving.
type ReduceLROnPlateau struct {
	opt       Optimizer
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
		opt:       opt,
		factor:    factor,
		threshold: threshold,
		patience:  patience,
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
		s.opt.SetLR(s.opt.LR() * s.factor)
		s.numBad = 0
	}
}

// PolynomialLR decays the lr as initial_lr * (1 - step/totalIters)^power.
// Once `count` reaches totalIters the lr is clamped at 0.
type PolynomialLR struct {
	opt        Optimizer
	totalIters int
	power      float64
	baseLR     float64
	count      int
}

// NewPolynomialLR constructs a PolynomialLR scheduler.
func NewPolynomialLR(opt Optimizer, totalIters int, power float64) *PolynomialLR {
	return &PolynomialLR{
		opt:        opt,
		totalIters: totalIters,
		power:      power,
		baseLR:     opt.LR(),
	}
}

// Step advances the scheduler one tick.
func (s *PolynomialLR) Step() {
	s.count++
	if s.totalIters <= 0 {
		return
	}
	if s.count >= s.totalIters {
		s.opt.SetLR(0)
		return
	}
	frac := 1 - float64(s.count)/float64(s.totalIters)
	s.opt.SetLR(s.baseLR * math.Pow(frac, s.power))
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

// CyclicLR follows a triangular schedule between baseLR and maxLR. Each half
// cycle lasts stepSize calls: the lr ramps up for stepSize steps, then down
// for stepSize steps, then repeats.
type CyclicLR struct {
	opt      Optimizer
	baseLR   float64
	maxLR    float64
	stepSize int
	count    int
}

// NewCyclicLR constructs a CyclicLR scheduler.
func NewCyclicLR(opt Optimizer, baseLR, maxLR float64, stepSize int) *CyclicLR {
	c := &CyclicLR{
		opt:      opt,
		baseLR:   baseLR,
		maxLR:    maxLR,
		stepSize: stepSize,
	}
	c.opt.SetLR(baseLR)
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
	s.opt.SetLR(lr)
}

// OneCycleLR is a simple triangular one-cycle schedule that ramps the lr
// from the optimizer's initial lr up to maxLR at the midpoint of totalSteps
// and back down to the initial lr.
type OneCycleLR struct {
	opt        Optimizer
	maxLR      float64
	totalSteps int
	baseLR     float64
	count      int
}

// NewOneCycleLR constructs a OneCycleLR scheduler.
func NewOneCycleLR(opt Optimizer, maxLR float64, totalSteps int) *OneCycleLR {
	return &OneCycleLR{
		opt:        opt,
		maxLR:      maxLR,
		totalSteps: totalSteps,
		baseLR:     opt.LR(),
	}
}

// Step advances the scheduler one tick.
func (s *OneCycleLR) Step() {
	s.count++
	if s.totalSteps <= 0 {
		return
	}
	half := float64(s.totalSteps) / 2
	t := float64(s.count)
	var lr float64
	if t <= half {
		// Ramp up: baseLR -> maxLR
		frac := t / half
		lr = s.baseLR + (s.maxLR-s.baseLR)*frac
	} else if t <= float64(s.totalSteps) {
		// Ramp down: maxLR -> baseLR
		frac := (t - half) / half
		lr = s.maxLR + (s.baseLR-s.maxLR)*frac
	} else {
		lr = s.baseLR
	}
	s.opt.SetLR(lr)
}
