package linear

import (
	"fmt"
	"time"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/maths"
	"github.com/bleak-and-bare/machine_learning/internal/maths/optimization"
	"golang.org/x/exp/constraints"
)

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

func (m *LinearRegression[T]) Fit(ds *dataset.DataSet[T]) error {
	start := time.Now()
	defer func() {
		elapsed := time.Since(start)
		fmt.Printf("LinearRegression fit took %v\n", elapsed)
	}()

	sgd := optimization.NewSGD[T](m.Threshold)
	sgd.Alpha = m.Alpha
	sgd.Cost = &maths.MSE[T]{
		Hypothesis: &linear_reg_hypo[T]{},
	}

	if err := sgd.Fit(ds); err != nil {
		return err
	}

	m.theta = sgd.GetParams()

	return nil
}
