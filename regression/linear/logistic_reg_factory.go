package linear

import (
	"github.com/bleak-and-bare/machine_learning/internal/maths/regularization"
	"github.com/bleak-and-bare/machine_learning/internal/selector"
	"golang.org/x/exp/constraints"
)

func LogRegFactory[T constraints.Float](blueprint LogisticRegression[T], params_map func(m map[string]float64) regularization.ElasticnetParams) selector.ModelFactory[T] {
	return func(hyper_params map[string]float64) selector.Model[T] {
		m := blueprint
		params := params_map(hyper_params)

		m.theta = nil
		m.scaffolded = true
		m.HyperParams = &params
		return &m
	}
}
