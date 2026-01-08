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

type linear_reg_hypo[T constraints.Float] struct{}

func (h *linear_reg_hypo[T]) On(params []T, sample *dataset.DataSample[T]) (T, error) {
	d, err := sample.DotProduct(params[1:])
	if err != nil {
		return 0.0, err
	}

	return params[0] + d, nil
}

func (h *linear_reg_hypo[T]) Diff(j int, params []T, sample *dataset.DataSample[T]) (T, error) {
	if j == 0 {
		return 1, nil
	}

	x := sample.GetFeat(j - 1)
	if x == nil {
		return 0.0, fmt.Errorf("No feature found at <%d, %d>", sample.GetRow(), j-1)
	}

	return *x, nil
}

func linear_reg_cost_partial_diff[T constraints.Float](j int, theta []T, ds *dataset.DataSet[T]) (T, error) {
	var h linear_reg_hypo[T]
	return optimization.PartialDiffMSE(j, theta, ds, &h)
}

func (m *LinearRegression[T]) Fit(ds *dataset.DataSet[T]) error {
	sgd := optimization.NewSGD[T]()
	sgd.Epsilon = m.Epsilon
	sgd.Alpha = m.Alpha
	sgd.MaxEpochs = m.MaxEpochs
	sgd.CostPartialDiff = linear_reg_cost_partial_diff

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
