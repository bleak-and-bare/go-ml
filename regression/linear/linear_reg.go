package linear

import (
	"fmt"
	"slices"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/iterable"
	"github.com/bleak-and-bare/machine_learning/internal/maths"
	"github.com/bleak-and-bare/machine_learning/internal/maths/vector"
	"github.com/bleak-and-bare/machine_learning/regression"
	"golang.org/x/exp/constraints"
)

type LinearRegression[T constraints.Float] struct {
	BaseModel[T]
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

func linear_reg_hypo_func[T constraints.Float](theta []T, x []T) T {
	return vector.DotProduct(iterable.Prepend(slices.Values(x), 1.0), slices.Values(theta))
}

func (m *LinearRegression[T]) PredictOn(ds *dataset.DataSet[T]) regression.RegressionReport[T] {
	pred := m.BaseModel.PredictOn(ds, linear_reg_hypo_func)
	return regression.ComputeReport(pred, ds.CollectTargets())
}

func (m *LinearRegression[T]) Predict(x []T) (T, error) {
	return m.BaseModel.Predict(x, linear_reg_hypo_func)
}
