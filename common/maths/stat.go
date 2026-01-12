package maths

import (
	"iter"
	"math"

	"golang.org/x/exp/constraints"
)

type Number interface {
	constraints.Integer | constraints.Float
}

func Mean[T Number](iter iter.Seq[T]) T {
	var sum T
	count := 0

	for v := range iter {
		sum += v
		count++
	}

	return sum / T(count)
}

func WelfordVar[T Number](iter iter.Seq[T]) (T, T, int) {
	n := 0
	var mean T
	var M2 T

	for v := range iter {
		n++
		diff := v - mean
		mean += diff / T(n)
		M2 += diff * (v - mean)
	}

	if n == 0 {
		return 0.0, 0.0, 0
	}

	return M2 / T(n), mean, n
}

func Variance[T Number](iter iter.Seq[T]) T {
	v, _, _ := WelfordVar(iter)
	return v
}

func Stdev[T Number](iter iter.Seq[T]) T {
	return T(math.Sqrt(float64(Variance(iter))))
}
