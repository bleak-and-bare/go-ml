package maths

import (
	"iter"
	"math"

	"github.com/bleak-and-bare/go-ml/machine_learning/common/iterable/accumulator"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/iterable/adapter"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/maths/vector"
	"golang.org/x/exp/constraints"
)

func L2Dist[T constraints.Float](v, w iter.Seq[T]) T {
	return L2Norm(vector.Diff(v, w))
}

func L1Norm[T constraints.Float](v iter.Seq[T]) T {
	return accumulator.Sum(adapter.Absolute(v))
}

func L2Norm[T constraints.Float](v iter.Seq[T]) T {
	return T(math.Sqrt(float64(accumulator.Sum(adapter.Squared(v)))))
}

func SquaredL2[T constraints.Float](v iter.Seq[T]) T {
	return accumulator.Sum(adapter.Squared(v))
}
