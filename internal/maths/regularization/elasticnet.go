package regularization

import (
	"slices"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/maths"
	"github.com/bleak-and-bare/machine_learning/internal/maths/utils"
	"golang.org/x/exp/constraints"
)

type ElasticnetParams struct {
	Lambda float64 // Regularization strength
	Alpha  float64 // L1/Lasso ratio
}

// Elasticnet implements BatchFunction interface
type Elasticnet[T constraints.Float] struct {
	ElasticnetParams
}

type ElasticnetParamsGrid struct {
	Alpha  []float64
	Lambda []float64
}

func ElasticnetDefGrid() ElasticnetParamsGrid {
	return ElasticnetParamsGrid{
		Alpha:  utils.Linspace(0.1, 0.9, 3),
		Lambda: utils.Logspace(-2, 2, 5),
	}
}

func NewElasticnet[T constraints.Float](lambda float64, alpha float64) Elasticnet[T] {
	return Elasticnet[T]{
		ElasticnetParams{lambda, alpha},
	}
}

func NewLasso[T constraints.Float](lambda float64) Elasticnet[T] {
	return Elasticnet[T]{
		ElasticnetParams{lambda, 1.0},
	}
}

func NewRidge[T constraints.Float](lambda float64) Elasticnet[T] {
	return Elasticnet[T]{
		ElasticnetParams{lambda, 0.0},
	}
}

func (e *Elasticnet[T]) On(params []T, _ *dataset.DataSet[T]) (T, error) {
	return T(e.Lambda) * (T(e.Alpha)*maths.L1Norm(slices.Values(params[1:])) + T(0.5*(1-e.Alpha))*maths.L2Norm(slices.Values(params[1:]))), nil
}

func (e *Elasticnet[T]) Diff(j int, params []T, _ *dataset.DataSet[T]) (T, error) {
	if j == 0 {
		return 0.0, nil
	}

	return T(e.Lambda*(1-e.Alpha))*params[j] + T(e.Lambda*e.Alpha*float64(utils.Sign(params[j]))), nil
}
