package metrics

import (
	"iter"
	"math"

	"github.com/bleak-and-bare/machine_learning/internal/iterable"
	"github.com/bleak-and-bare/machine_learning/internal/iterable/accumulator"
	"github.com/bleak-and-bare/machine_learning/internal/iterable/adapter"
	"github.com/bleak-and-bare/machine_learning/internal/maths"
	"github.com/bleak-and-bare/machine_learning/internal/maths/vector"
	"golang.org/x/exp/constraints"
)

type Number interface {
	constraints.Float | constraints.Integer
}

func R2Score[T Number](trg, pred iter.Seq[T]) float64 {
	trg_var, _, n := maths.WelfordVar(trg)
	return float64(1.0 - accumulator.Sum(adapter.Squared(vector.Diff(trg, pred)))/(trg_var*T(n)))
}

// Root mean squared error
func RMSE[T Number](trg, pred iter.Seq[T]) float64 {
	return math.Sqrt(float64(maths.Mean(adapter.Squared(vector.Diff(trg, pred)))))
}

// Mean absolute error
func MAE[T Number](trg, pred iter.Seq[T]) float64 {
	return float64(maths.Mean(iterable.Map(vector.Diff(trg, pred), func(v T) T {
		return T(math.Abs(float64(v)))
	})))
}
