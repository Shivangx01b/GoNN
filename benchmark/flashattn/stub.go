//go:build !cuda
// +build !cuda

package main

import "fmt"

func main() {
	fmt.Println("benchmark/flashattn requires the CUDA build: go run -tags cuda ./benchmark/flashattn")
}
