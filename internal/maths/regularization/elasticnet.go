package regularization

import (
	"slices"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/maths"
	"github.com/bleak-and-bare/machine_learning/internal/maths/utils"
	"golang.org/x/exp/constraints"
)

// Elasticnet implements BatchFunction interface
type Elasticnet[T constraints.Float] struct {
	Lambda float64 // Regularization strength
	Alpha  float64 // L1/Lasso ratio
}

type HyperParamsGrid struct {
	Alpha  []float64
	Lambda []float64
}

func ElasticnetDefGrid() HyperParamsGrid {
	return HyperParamsGrid{
		Alpha:  []float64{0.1, 0.3, 0.5, 0.7, 0.9},
		Lambda: utils.Logspace(-4, 2, 7),
	}
}

func NewLasso[T constraints.Float](lambda float64) Elasticnet[T] {
	return Elasticnet[T]{
		Lambda: lambda,
		Alpha:  1.0,
	}
}

func NewRidge[T constraints.Float](lambda float64) Elasticnet[T] {
	return Elasticnet[T]{
		Lambda: 2 * lambda,
		Alpha:  0.0,
	}
}

func (e *Elasticnet[T]) On(params []T, _ *dataset.DataSet[T]) (T, error) {
	return T(e.Lambda) * (T(e.Alpha)*maths.L1Norm(slices.Values(params)) + T(0.5*(1-e.Alpha))*maths.L2Norm(slices.Values(params))), nil
}

func (e *Elasticnet[T]) Diff(j int, params []T, _ *dataset.DataSet[T]) (T, error) {
	if j == 0 {
		return 0.0, nil
	}

	return T(2*e.Lambda*(1-e.Alpha))*params[j] + T(e.Lambda*e.Alpha*float64(utils.Sign(params[j]))), nil
}
