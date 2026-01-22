package selector

import (
	"math"
	"slices"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/maths/stat"
	"github.com/bleak-and-bare/machine_learning/internal/maths/utils"
	"golang.org/x/exp/constraints"
)

// Choose hyper parameter based on estimator using cross validation
type GridSearch[T constraints.Float] struct {
	param_choices [][]float64
	model_factory ModelFactory[T]
	kfold         int
	best_params   []float64
}

func NewGridSearch[T constraints.Float](param_choices [][]float64, model_factory ModelFactory[T]) GridSearch[T] {
	return GridSearch[T]{
		param_choices,
		model_factory,
		5,
		nil,
	}
}

func (g *GridSearch[T]) BestParams() []float64 {
	return g.best_params
}

func (g *GridSearch[T]) Fit(ds *dataset.DataSet[T]) error {
	combinations := utils.CartesianProduct(g.param_choices)
	best_loss := math.Inf(1)
	var best_params []float64

	for _, comb := range combinations {
		i := 0
		cv_losses := make([]T, g.kfold)

		for fold := range ds.KFoldSplit(g.kfold) {
			m := g.model_factory(comb)
			if err := m.Fit(fold.Train); err != nil {
				return err
			}

			loss, err := m.Loss(fold.Test)
			if err != nil {
				return err
			}

			cv_losses[i] = loss
			i++
		}

		loss := stat.Mean(slices.Values(cv_losses))
		if loss < T(best_loss) {
			best_loss = float64(loss)
			best_params = comb
		}
	}

	g.best_params = best_params

	return nil
}
