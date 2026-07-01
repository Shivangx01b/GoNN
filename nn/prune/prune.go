// Package prune implements mask-based parameter pruning shaped after
// torch.nn.utils.prune, adapted to a hook-free design.
//
// PyTorch installs a forward pre-hook that recomputes weight = weight_orig *
// weight_mask before every forward, so pruned entries stay zero even as the
// optimizer updates weight_orig. Go modules here have no hooks, so this
// package prunes the parameter tensor in place and keeps the original values
// and the mask in a package registry keyed by tensor identity:
//
//   - Prune / the method helpers (L1Unstructured, ...) zero the masked
//     entries of p.Data immediately.
//   - Gradients flow into the parameter as usual; an optimizer step (weight
//     decay, momentum, ...) can move masked entries off zero. Call
//     Reapply(params...) after each optimizer step to re-zero them — this is
//     the explicit replacement for PyTorch's forward pre-hook.
//   - Remove(p) makes pruning permanent: the registry entry (including the
//     snapshot of the original values) is dropped and the current, masked
//     data is kept, like torch.nn.utils.prune.remove.
//
// Amount semantics (deviation, documented): amounts are fractions in [0, 1]
// only; PyTorch's integer "number of connections" form is not supported. The
// number of pruned entries is round(amount * n), matching PyTorch's rounding
// of fractional amounts.
//
// Iterative pruning composes like PyTorch's PruningContainer: pruning an
// already-pruned tensor multiplies the new mask into the existing one, and
// importance scores (L1, Ln) are computed on the current, already-masked
// values.
package prune

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"

	"gonn/tensor"
)

type entry struct {
	orig []float64 // snapshot of the values at first prune
	mask []float64 // 0 = pruned, 1 = kept
}

var (
	mu       sync.Mutex
	registry = map[*tensor.Tensor]*entry{}
)

// Prune applies mask to p: masked entries of p.Data are zeroed in place and
// the mask is remembered so Reapply can re-zero them later. mask must have
// exactly len(p.Data) elements (0 = prune, anything non-zero = keep; values
// are normalized to {0, 1}). Pruning an already-pruned tensor combines the
// masks multiplicatively.
func Prune(p *tensor.Tensor, mask []float64) {
	if len(mask) != len(p.Data) {
		panic(fmt.Sprintf("prune: mask length %d does not match tensor numel %d",
			len(mask), len(p.Data)))
	}
	mu.Lock()
	defer mu.Unlock()
	e := registry[p]
	if e == nil {
		orig := make([]float64, len(p.Data))
		copy(orig, p.Data)
		e = &entry{orig: orig, mask: make([]float64, len(mask))}
		for i := range e.mask {
			e.mask[i] = 1
		}
		registry[p] = e
	}
	for i, m := range mask {
		if m == 0 {
			e.mask[i] = 0
			p.Data[i] = 0
		}
	}
}

// Reapply re-applies the stored masks to the given pruned tensors, zeroing
// any masked entry the optimizer moved off zero. Call it after every
// optimizer step; it replaces PyTorch's forward pre-hook. Tensors that were
// never pruned (or whose pruning was Remove'd) are skipped.
func Reapply(params ...*tensor.Tensor) {
	mu.Lock()
	defer mu.Unlock()
	for _, p := range params {
		e := registry[p]
		if e == nil {
			continue
		}
		for i, m := range e.mask {
			if m == 0 {
				p.Data[i] = 0
			}
		}
	}
}

// Remove makes pruning permanent for p: the registry entry (original values
// and mask) is dropped and the current, masked data is kept.
func Remove(p *tensor.Tensor) {
	mu.Lock()
	defer mu.Unlock()
	delete(registry, p)
}

// IsPruned reports whether p currently has a pruning mask registered.
func IsPruned(p *tensor.Tensor) bool {
	mu.Lock()
	defer mu.Unlock()
	return registry[p] != nil
}

// Mask returns a copy of p's current pruning mask (1 = kept, 0 = pruned), or
// nil if p is not pruned.
func Mask(p *tensor.Tensor) []float64 {
	mu.Lock()
	defer mu.Unlock()
	e := registry[p]
	if e == nil {
		return nil
	}
	out := make([]float64, len(e.mask))
	copy(out, e.mask)
	return out
}

// Orig returns a copy of the values p held when it was first pruned, or nil
// if p is not pruned.
func Orig(p *tensor.Tensor) []float64 {
	mu.Lock()
	defer mu.Unlock()
	e := registry[p]
	if e == nil {
		return nil
	}
	out := make([]float64, len(e.orig))
	copy(out, e.orig)
	return out
}

// nToPrune converts a fractional amount into a count, PyTorch style:
// round(amount * n). Panics if amount is outside [0, 1].
func nToPrune(amount float64, n int) int {
	if amount < 0 || amount > 1 {
		panic(fmt.Sprintf("prune: amount must be a fraction in [0, 1], got %g", amount))
	}
	k := int(math.Round(amount * float64(n)))
	if k > n {
		k = n
	}
	return k
}

// onesMask returns an all-ones mask of length n.
func onesMask(n int) []float64 {
	m := make([]float64, n)
	for i := range m {
		m[i] = 1
	}
	return m
}

// Identity registers a no-op (all-ones) pruning mask on p, mirroring
// torch.nn.utils.prune.identity. Useful as a starting point for iterative
// pruning.
func Identity(p *tensor.Tensor) {
	Prune(p, onesMask(len(p.Data)))
}

// CustomFromMask prunes p with a user-provided mask, mirroring
// torch.nn.utils.prune.custom_from_mask.
func CustomFromMask(p *tensor.Tensor, mask []float64) {
	Prune(p, mask)
}

// RandomUnstructured prunes round(amount*n) randomly chosen elements of p.
// The choice is deterministic for a given seed.
func RandomUnstructured(p *tensor.Tensor, amount float64, seed int64) {
	n := len(p.Data)
	k := nToPrune(amount, n)
	mask := onesMask(n)
	rng := rand.New(rand.NewSource(seed))
	for _, idx := range rng.Perm(n)[:k] {
		mask[idx] = 0
	}
	Prune(p, mask)
}

// L1Unstructured prunes the round(amount*n) elements of p with the smallest
// absolute value (computed on the current, possibly already-masked data).
func L1Unstructured(p *tensor.Tensor, amount float64) {
	n := len(p.Data)
	k := nToPrune(amount, n)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		return math.Abs(p.Data[idx[a]]) < math.Abs(p.Data[idx[b]])
	})
	mask := onesMask(n)
	for _, i := range idx[:k] {
		mask[i] = 0
	}
	Prune(p, mask)
}

// sliceGeometry decomposes p's shape around dim: p viewed as
// (outer, size, inner) with the pruning dimension in the middle.
func sliceGeometry(p *tensor.Tensor, dim int) (outer, size, inner int) {
	rank := len(p.Shape)
	if dim < 0 {
		dim += rank
	}
	if dim < 0 || dim >= rank {
		panic(fmt.Sprintf("prune: dim %d out of range for shape %v", dim, p.Shape))
	}
	outer, inner = 1, 1
	for i := 0; i < dim; i++ {
		outer *= p.Shape[i]
	}
	size = p.Shape[dim]
	for i := dim + 1; i < rank; i++ {
		inner *= p.Shape[i]
	}
	return outer, size, inner
}

// structuredMask builds a mask zeroing the whole slices along dim listed in
// pruneSlices.
func structuredMask(p *tensor.Tensor, dim int, pruneSlices []int) []float64 {
	outer, size, inner := sliceGeometry(p, dim)
	mask := onesMask(len(p.Data))
	drop := make(map[int]bool, len(pruneSlices))
	for _, s := range pruneSlices {
		drop[s] = true
	}
	for o := 0; o < outer; o++ {
		for s := 0; s < size; s++ {
			if !drop[s] {
				continue
			}
			base := (o*size + s) * inner
			for i := 0; i < inner; i++ {
				mask[base+i] = 0
			}
		}
	}
	return mask
}

// LnStructured prunes whole slices of p along dim: the round(amount*size)
// slices with the smallest Ln norm (n = 1, 2, ... or math.Inf(1) for the max
// norm) are zeroed entirely, mirroring torch.nn.utils.prune.ln_structured.
func LnStructured(p *tensor.Tensor, amount float64, n float64, dim int) {
	outer, size, inner := sliceGeometry(p, dim)
	k := nToPrune(amount, size)
	norms := make([]float64, size)
	for s := 0; s < size; s++ {
		acc := 0.0
		for o := 0; o < outer; o++ {
			base := (o*size + s) * inner
			for i := 0; i < inner; i++ {
				a := math.Abs(p.Data[base+i])
				if math.IsInf(n, 1) {
					if a > acc {
						acc = a
					}
				} else {
					acc += math.Pow(a, n)
				}
			}
		}
		if math.IsInf(n, 1) {
			norms[s] = acc
		} else {
			norms[s] = math.Pow(acc, 1/n)
		}
	}
	idx := make([]int, size)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return norms[idx[a]] < norms[idx[b]] })
	Prune(p, structuredMask(p, dim, idx[:k]))
}

// RandomStructured prunes round(amount*size) randomly chosen whole slices of
// p along dim. Deterministic for a given seed.
func RandomStructured(p *tensor.Tensor, amount float64, dim int, seed int64) {
	_, size, _ := sliceGeometry(p, dim)
	k := nToPrune(amount, size)
	rng := rand.New(rand.NewSource(seed))
	Prune(p, structuredMask(p, dim, rng.Perm(size)[:k]))
}

// GlobalUnstructured prunes the round(amount*total) elements with the
// smallest absolute value across all the given tensors together (global L1
// threshold), mirroring torch.nn.utils.prune.global_unstructured with
// L1Unstructured as the method.
func GlobalUnstructured(params []*tensor.Tensor, amount float64) {
	total := 0
	for _, p := range params {
		total += len(p.Data)
	}
	k := nToPrune(amount, total)

	type ref struct {
		param int
		idx   int
	}
	refs := make([]ref, 0, total)
	for pi, p := range params {
		for i := range p.Data {
			refs = append(refs, ref{param: pi, idx: i})
		}
	}
	sort.SliceStable(refs, func(a, b int) bool {
		return math.Abs(params[refs[a].param].Data[refs[a].idx]) <
			math.Abs(params[refs[b].param].Data[refs[b].idx])
	})

	masks := make([][]float64, len(params))
	for pi, p := range params {
		masks[pi] = onesMask(len(p.Data))
	}
	for _, r := range refs[:k] {
		masks[r.param][r.idx] = 0
	}
	for pi, p := range params {
		Prune(p, masks[pi])
	}
}
