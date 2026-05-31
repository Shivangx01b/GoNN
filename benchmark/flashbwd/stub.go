//go:build !cuda
// +build !cuda

package main

import "fmt"

func main() {
	fmt.Println("benchmark/flashbwd requires the CUDA build: go run -tags cuda ./benchmark/flashbwd")
}
