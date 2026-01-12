package maths

import (
	"iter"
	"math"

	"github.com/bleak-and-bare/machine_learning/internal/iterable/accumulator"
	"github.com/bleak-and-bare/machine_learning/internal/iterable/adapter"
	"github.com/bleak-and-bare/machine_learning/internal/iterable/vector"
	"golang.org/x/exp/constraints"
)

func L2Dist[T constraints.Float](v, w iter.Seq[T]) T {
	return L2Norm(vector.Diff(v, w))
}

func L2Norm[T constraints.Float](v iter.Seq[T]) T {
	return T(math.Sqrt(float64(accumulator.Sum(adapter.Squared(v)))))
}
