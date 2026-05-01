package main

import (
	"fmt"

	"github.com/rcarmo/go-tinygrad/tensor"
)

func main() {
	fmt.Println("go-tinygrad demo")
	fmt.Println("================")

	// Basic tensor ops
	a := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	b := tensor.FromFloat32([]float32{7, 8, 9, 10, 11, 12}, []int{2, 3})

	fmt.Println("\na =", a.Data())
	fmt.Println("b =", b.Data())

	c := a.Add(b).Mul(a)
	fmt.Println("\n(a + b) * a =", c.Data())

	// MatMul
	x := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	w := tensor.FromFloat32([]float32{1, 0, 0, 0, 1, 0, 0, 0, 1}, []int{3, 3})
	y := x.MatMul(w)
	fmt.Println("\nMatMul [2,3] @ [3,3] =", y.Data(), "shape:", y.Shape())

	// Softmax
	logits := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int{1, 4})
	probs := logits.Softmax()
	fmt.Println("\nSoftmax([1,2,3,4]) =", probs.Data())

	// Layer norm
	data := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	gamma := tensor.Ones([]int{3})
	beta := tensor.Zeros([]int{3})
	normed := data.LayerNorm(gamma, beta, 1e-5)
	fmt.Println("\nLayerNorm([[1,2,3],[4,5,6]]) =", normed.Data())

	// Reduce
	r := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3})
	fmt.Println("\nSum(axis=1) =", r.Sum(1).Data())
	fmt.Println("Max(axis=0) =", r.ReduceMax(0).Data())

	// GELU
	g := tensor.FromFloat32([]float32{-2, -1, 0, 1, 2}, []int{5})
	fmt.Println("\nGELU([-2,-1,0,1,2]) =", g.GELU().Data())
}
