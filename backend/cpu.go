package backend

type cpuBackend struct{}

func (cpuBackend) Name() Device { return CPU }

func (cpuBackend) MatMul(a, b []float64, m, k, n int) []float64 {
	out := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for kk := 0; kk < k; kk++ {
			x := a[i*k+kk]
			if x == 0 {
				continue
			}
			for j := 0; j < n; j++ {
				out[i*n+j] += x * b[kk*n+j]
			}
		}
	}
	return out
}

func (cpuBackend) AddElem(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func (cpuBackend) MulElem(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] * b[i]
	}
	return out
}

func (cpuBackend) Synchronize() {}
