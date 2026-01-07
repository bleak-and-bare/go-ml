package linear

import (
	"errors"
	"fmt"
	"math"

	"github.com/bleak-and-bare/machine_learning/common/dataset"
	"github.com/bleak-and-bare/machine_learning/common/maths/optimization"
	"github.com/bleak-and-bare/machine_learning/regression"
	"golang.org/x/exp/constraints"
)

type LinearRegression[T constraints.Float] struct {
	theta     []T     // parameter list
	Alpha     float32 // learning rate
	Epsilon   float32
	MaxEpochs uint32
}

func NewLinearReg[T constraints.Float]() LinearRegression[T] {
	return LinearRegression[T]{
		Alpha:     1e-2,
		Epsilon:   1e-3,
		MaxEpochs: 10_000,
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

func (m *LinearRegression[T]) Fit(ds *dataset.DataSet[T]) error {
	sgd := optimization.NewSGD[T]()
	sgd.Alpha = m.Alpha
	sgd.Epsilon = m.Epsilon
	sgd.MaxEpochs = m.MaxEpochs

	if err := sgd.Fit(ds); err != nil {
		return err
	}

	m.theta = sgd.GetParams()

	return nil
}

func (m *LinearRegression[T]) PredictOn(ds *dataset.DataSet[T]) regression.RegressionReport[T] {
	r := regression.RegressionReport[T]{
		DataSet: ds,
	}

	ds.ForEachSample(func(ds dataset.DataSample[T]) bool {
		sample, err := ds.GetSampleTest()
		if err != nil {
			r.SkippedRows++
			return true
		}

		pred, err := m.Predict(sample)
		if err != nil {
			r.SkippedRows++
			return true
		}

		if y := ds.GetTarget(); y != nil {
			diff := *y - pred
			r.Predictions = append(r.Predictions, pred)
			r.MeanAbsoluteErr += math.Abs(float64(diff))
			r.RootMeanSquareErr += float64(diff * diff)
		} else {
			r.SkippedRows++
		}

		return true
	})

	count := ds.Size() - uint32(r.SkippedRows)
	if count > 0 {
		var_estimator := r.RootMeanSquareErr / float64(count)
		r.Score = 1 - var_estimator/ds.TargetVariance()
		r.RootMeanSquareErr = math.Sqrt(r.RootMeanSquareErr) / float64(count)
		r.MeanAbsoluteErr /= float64(count)
	}

	return r
}

func (m *LinearRegression[T]) Predict(x []T) (T, error) {
	if len(m.theta) == 0 {
		return 0.0, errors.New("Using non-fit model")
	}

	if len(x) != len(m.theta)-1 {
		return 0.0, errors.New("Invalid vector provided")
	}

	sum := m.theta[0]
	for i := range x {
		sum += x[i] * m.theta[i+1]
	}

	return sum, nil
}
