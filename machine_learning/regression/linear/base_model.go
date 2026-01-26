package linear

import (
	"errors"

	"github.com/bleak-and-bare/go-ml/machine_learning/common/dataset"
	"github.com/bleak-and-bare/go-ml/machine_learning/regression"
	"golang.org/x/exp/constraints"
)

type BaseModel[T constraints.Float] struct {
	theta []T // parameter list
}

func (m *BaseModel[T]) PredictOn(ds *dataset.DataSet[T], hypothesis func(theta []T, x []T) T) []T {
	r := regression.RegressionReport[T]{
		DataSet: ds,
	}

	targets := make([]T, 0, ds.Size())
	predictions := make([]T, 0, ds.Size())
	for ds := range ds.Samples() {
		sample, err := ds.GetSampleTest()
		if err != nil {
			r.SkippedRows++
			continue
		}

		pred, err := m.Predict(sample, hypothesis)
		if err != nil {
			r.SkippedRows++
			continue
		}

		if y := ds.GetTarget(); y != nil {
			targets = append(targets, *y)
			predictions = append(predictions, pred)
		} else {
			r.SkippedRows++
		}
	}

	return predictions
}

func (m *BaseModel[T]) Predict(x []T, hypothesis func(theta []T, x []T) T) (T, error) {
	if len(m.theta) == 0 {
		return 0.0, errors.New("Using non-fit model")
	}

	if len(x) != len(m.theta)-1 {
		return 0.0, errors.New("Invalid vector provided")
	}

	return hypothesis(m.theta, x), nil
}
