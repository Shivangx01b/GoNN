//go:build !opencl
// +build !opencl

package main

import "fmt"

func main() {
	fmt.Println("benchmark/openclcheck requires the OpenCL build: go run -tags opencl ./benchmark/openclcheck")
}
