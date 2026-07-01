package tensor

// Reverse-mode automatic differentiation: the Function graph node, the
// Backward pass (iterative topological sort — safe for arbitrarily deep
// graphs such as long RNN unrolls), and the MakeNode escape hatch for
// custom ops.

// Function records the op that produced a tensor and how to backprop.
type Function struct {
	Name     string
	Inputs   []*Tensor
	Saved    []interface{}
	Backward func(grad *Tensor, saved []interface{}, inputs []*Tensor) []*Tensor
}

// Backward computes gradients by walking the autograd DAG in reverse.
// t must be a scalar (or you must call .Sum() first).
func (t *Tensor) Backward() {
	if len(t.Data) != 1 {
		opError("Backward", "can only call on scalar tensors; use t.Sum().Backward()")
	}
	order := topoOrder(t)

	// Seed gradient at root.
	if t.Grad == nil {
		t.Grad = Ones(t.Shape...)
	} else {
		for i := range t.Grad.Data {
			t.Grad.Data[i] = 1
		}
	}

	// Walk in reverse, pushing grads to inputs.
	for i := len(order) - 1; i >= 0; i-- {
		n := order[i]
		if n.creator == nil {
			continue
		}
		grads := n.creator.Backward(n.Grad, n.creator.Saved, n.creator.Inputs)
		for j, p := range n.creator.Inputs {
			if !p.RequiresGrad && p.creator == nil {
				continue
			}
			if grads[j] == nil {
				continue
			}
			g := grads[j]
			// Sum-reduce gradient back to p's shape if broadcasting expanded it.
			g = unbroadcast(g, p.Shape)
			if p.Grad == nil {
				p.Grad = g
			} else {
				for k := range p.Grad.Data {
					p.Grad.Data[k] += g.Data[k]
				}
			}
		}
	}
}

// topoOrder returns the tensors reachable from root in topological order
// (parents before children). Iterative post-order DFS with an explicit
// stack: recursion depth is O(1) regardless of graph depth, so very deep
// chains (long unrolled sequences) cannot overflow the goroutine stack.
func topoOrder(root *Tensor) []*Tensor {
	type frame struct {
		n        *Tensor
		expanded bool
	}
	visited := map[*Tensor]bool{}
	order := []*Tensor{}
	stack := []frame{{root, false}}
	for len(stack) > 0 {
		f := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if f.expanded {
			order = append(order, f.n)
			continue
		}
		if visited[f.n] {
			continue
		}
		visited[f.n] = true
		stack = append(stack, frame{f.n, true})
		if f.n.creator != nil {
			// Push in reverse so inputs pop in declaration order, matching the
			// recursive DFS this replaced (keeps gradient accumulation order —
			// and therefore floating-point results — bit-identical).
			ins := f.n.creator.Inputs
			for i := len(ins) - 1; i >= 0; i-- {
				if p := ins[i]; p != nil && !visited[p] {
					stack = append(stack, frame{p, false})
				}
			}
		}
	}
	return order
}

// MakeNode attaches a custom autograd node to out: it records inputs and a
// backward closure that, given out's gradient, returns the gradient for each
// input (same order as inputs; a nil entry skips that input). This is the
// escape hatch for custom ops (e.g. a fused CUDA kernel) that compute their
// own forward/backward outside the built-in op set. If no input requires grad,
// out is left as a plain leaf.
func MakeNode(out *Tensor, name string, inputs []*Tensor, backward func(grad *Tensor) []*Tensor) {
	needsGrad := false
	for _, in := range inputs {
		if in != nil && (in.RequiresGrad || in.creator != nil) {
			needsGrad = true
			break
		}
	}
	if !needsGrad {
		return
	}
	out.RequiresGrad = true
	out.creator = &Function{
		Name:   name,
		Inputs: inputs,
		Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
			return backward(grad)
		},
	}
}
