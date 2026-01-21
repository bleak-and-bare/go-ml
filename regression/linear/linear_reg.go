package linear

import (
	"errors"
	"fmt"
	"slices"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/iterable"
	"github.com/bleak-and-bare/machine_learning/internal/maths"
	"github.com/bleak-and-bare/machine_learning/internal/maths/metrics"
	"github.com/bleak-and-bare/machine_learning/internal/maths/vector"
	"github.com/bleak-and-bare/machine_learning/regression"
	"golang.org/x/exp/constraints"
)

type LinearRegression[T constraints.Float] struct {
	theta     []T     // parameter list
	Alpha     float32 // learning rate
	Threshold maths.Threshold
}

func NewLinearReg[T constraints.Float]() LinearRegression[T] {
	return LinearRegression[T]{
		Alpha:     1e-4,
		Threshold: maths.DefThreshold(),
	}
}

func (m *LinearRegression[T]) PrintRegLineEquation() {
	fmt.Printf("y = ")
	for i, theta := range m.theta {
		if i == 0 {
			fmt.Printf("%.3f", theta)
		} else if len(m.theta) > 2 {
			fmt.Printf(" + %.3f*x%d", theta, i)
		} else {
			fmt.Printf(" + %.3f*x", theta)
		}
	}
	fmt.Println("")
}

func (m *LinearRegression[T]) PredictOn(ds *dataset.DataSet[T]) regression.RegressionReport[T] {
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

		pred, err := m.Predict(sample)
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

	r.Predictions = predictions
	r.Score = metrics.R2Score(slices.Values(targets), slices.Values(predictions))
	r.RootMeanSquareErr = metrics.RMSE(slices.Values(targets), slices.Values(predictions))
	r.MeanAbsoluteErr = metrics.MAE(slices.Values(targets), slices.Values(predictions))

	return r
}

func (m *LinearRegression[T]) Predict(x []T) (T, error) {
	if len(m.theta) == 0 {
		return 0.0, errors.New("Using non-fit model")
	}

	if len(x) != len(m.theta)-1 {
		return 0.0, errors.New("Invalid vector provided")
	}

	return vector.DotProduct(slices.Values(m.theta), iterable.Prepend(slices.Values(x), 1)), nil
}
