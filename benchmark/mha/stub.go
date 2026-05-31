//go:build !cuda
// +build !cuda

package main

import "fmt"

func main() {
	fmt.Println("benchmark/mha requires the CUDA build: go run -tags cuda ./benchmark/mha")
}
