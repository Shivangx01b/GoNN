package optim

// Weight-decay semantics per optimizer (kept exactly as historically
// implemented — unifying them would change trained results):
//
//	coupled L2 (g += wd*w, via coupledWD):
//	    SGD, Adam, RMSprop, Adagrad, Adadelta, NAdam, Adamax, RAdam, ASGD
//	decoupled (parameter scaled directly, inline in the element loop):
//	    AdamW (w -= lr*wd*w), Lion (folded into the sign update),
//	    LAMB (wd*w added to the update BEFORE the trust-ratio norm — cannot
//	    be expressed as a separate parameter-scale pass)
//	none (Group.WeightDecay is ignored; documented on the constructors):
//	    Rprop, SparseAdam, Adafactor, LBFGS

// coupledWD returns the gradient with coupled L2 weight decay applied:
// g + wd*w (no-op when wd == 0).
func coupledWD(g, w, wd float64) float64 {
	if wd != 0 {
		return g + wd*w
	}
	return g
}
