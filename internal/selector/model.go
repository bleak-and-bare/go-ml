package selector

import (
	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"golang.org/x/exp/constraints"
)

type ModelFactory[T constraints.Float] func(hyper_params map[string]float64) Model[T]

type Model[T constraints.Float] interface {
	Fit(*dataset.DataSet[T]) error
	Loss(*dataset.DataSet[T]) (T, error)
}
