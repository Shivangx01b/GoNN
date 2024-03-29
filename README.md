<h1 align="center">
  <br>
  <a href=""><img src="https://github.com/Shivangx01b/GoNN/blob/main/static/logo.png" alt="" width="200px;"></a>
  <br>
  <img src="https://img.shields.io/github/languages/top/Shivangx01b/GoNN?style=flat-square">
  <a href="https://goreportcard.com/report/github.com/Shivangx01b/GoNN"><img src="https://goreportcard.com/badge/github.com/Shivangx01b/GoNN"></a>
  <a href="https://twitter.com/intent/follow?screen_name=shivangx01b"><img src="https://img.shields.io/twitter/follow/shivangx01b?style=flat-square"></a>
</h1>

# GoNN
A go based neural network framework

## What is GONN ?

GoNN is Go Lang based Neural Network for deep learning (work in progress!). This project does not compete with any exsisting neural network library which is world famouse like Pytroch, Tensorflow etc; but still it's a deep learning framework (I feel so ?)

## Example usage

- Using GoNN
```go
package main

import (
	"fmt"
	"gonn/tensor"
)

type Tensor struct {
	data []float64
}

func main() {
	// Example of how to create a tensor and perform an operation
	xData := []float64{1.0, 2.0, 3.0}
	yData := []float64{4.0, 5.0, 6.0}
	xTensor := Tensor{data: xData}
	yTensor := Tensor{data: yData}
	mul := tensor.Mul{}
	ctx := tensor.Context{}
	result := mul.Forward(&ctx, xTensor.data, yTensor.data)
	fmt.Println("Result of multiplication:", result)
}
```

- Using Pytorch
```python
import torch

def main():
    # Example of how to create a tensor and perform an operation
    x_data = torch.tensor([1.0, 2.0, 3.0])
    y_data = torch.tensor([4.0, 5.0, 6.0])

    # Perform element-wise multiplication
    result = x_data * y_data

    print("Result of multiplication:", result)

if __name__ == "__main__":
    main()
```


### TODO 
1) Impelement Gonum for matrix multiplication
2) Impelemnt 2D and 3D vectors
3) Better Backwards capability (looks like shit for now)
4) Better error handling
5) Test MNIST dataset
