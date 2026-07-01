package tensor

import (
	"sort"
	"strings"
)

// The unary-op registry is the single source of truth for elementwise unary
// operations (activations and math functions): one UnaryOpDef supplies the
// CPU forward closure, the autograd backward closure, and the backend
// dispatch kind (GPU kernel) for an op. The fluent Tensor methods, the
// name-based (*Tensor).Unary entry point used by nn's generic Activation
// module, and GPU dispatch all derive from the same definition.
//
// Fixed (parameter-free) ops are registered by canonical lowercase name.
// Parameterized ops (LeakyReLU(alpha), Threshold(t, v), ...) construct
// ephemeral UnaryOpDefs inside their fluent methods — their closures capture
// the parameters, so they cannot live in a name-keyed table.

// UnaryOpDef describes one elementwise unary op.
type UnaryOpDef struct {
	// Name is the canonical lowercase identifier (e.g. "gelu") for fixed ops,
	// or a descriptive autograd node label for parameterized ones.
	Name string
	// Kind selects the accelerated backend kernel; UnaryNone means the op has
	// no kernel and always runs the Go closure.
	Kind UnaryKind
	// Fwd computes y = f(x) for one element.
	Fwd func(x float64) float64
	// Bwd computes the input gradient from the upstream gradient g, the input
	// x, and the forward output y.
	Bwd func(g, x, y float64) float64
}

var unaryRegistry = map[string]UnaryOpDef{}

// RegisterUnary adds a fixed unary op to the registry. The name is
// case-insensitive (stored lowercase). Panics on duplicates so a typo'd
// re-registration cannot silently shadow an op.
func RegisterUnary(d UnaryOpDef) {
	key := strings.ToLower(d.Name)
	if _, exists := unaryRegistry[key]; exists {
		opError("RegisterUnary", "op %q already registered", key)
	}
	if d.Fwd == nil || d.Bwd == nil {
		opError("RegisterUnary", "op %q must define Fwd and Bwd", key)
	}
	unaryRegistry[key] = d
}

// LookupUnary returns the registered op for name (case-insensitive).
func LookupUnary(name string) (UnaryOpDef, bool) {
	d, ok := unaryRegistry[strings.ToLower(name)]
	return d, ok
}

// UnaryOpNames returns the sorted canonical names of all registered ops.
func UnaryOpNames() []string {
	names := make([]string, 0, len(unaryRegistry))
	for k := range unaryRegistry {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// Unary applies the registered op by name (case-insensitive), with autograd
// and backend dispatch. Panics on unknown names — use LookupUnary to probe.
func (t *Tensor) Unary(name string) *Tensor {
	d, ok := LookupUnary(name)
	if !ok {
		opError("Unary", "unknown op %q (registered: %s)", name, strings.Join(UnaryOpNames(), ", "))
	}
	return applyUnary(t, d)
}

// applyUnary is the single execution engine for elementwise unary ops: it
// runs the forward through the backend when dispatch accepts (large tensor +
// capable backend), else the Go closure; stamps the dtype; and attaches the
// autograd node. The backward always runs the Go closure on the host — the
// autograd engine accumulates into host slices, and keeping backward on the
// reference implementation preserves gradcheck-ability regardless of which
// device ran the forward.
func applyUnary(t *Tensor, d UnaryOpDef) *Tensor {
	out := Zeros(t.Shape...)
	if !dispatchUnary(d.Kind, t.Data, out.Data) {
		for i, v := range t.Data {
			out.Data[i] = d.Fwd(v)
		}
	}
	finishOp(out, t.Dtype)
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		input := t
		outData := out.Data
		bwd := d.Bwd
		out.creator = &Function{
			Name:   d.Name,
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				g := Zeros(grad.Shape...)
				for i := range g.Data {
					g.Data[i] = bwd(grad.Data[i], input.Data[i], outData[i])
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}
