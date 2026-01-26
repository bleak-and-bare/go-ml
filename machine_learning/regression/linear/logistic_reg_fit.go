package linear

import (
	"fmt"
	"time"

	"github.com/bleak-and-bare/go-ml/machine_learning/common/dataset"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/iterable"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/iterable/accumulator"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/maths"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/maths/optimization"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/maths/regularization"
	"golang.org/x/exp/constraints"
)

func (m *LogisticRegression[T]) Fit(ds *dataset.DataSet[T]) error {
	if m.Penalty != regularization.None && m.HyperParams == nil {
		return fmt.Errorf("LogisticRegression : no hyper parameters to use for regularizer")
	}

	if !m.scaffolded {
		start := time.Now()
		defer func() {
			elapsed := time.Since(start)
			fmt.Printf("LogisticRegression fit took %v\n", elapsed)
		}()
	}

	sgd := optimization.NewSGD[T](m.Threshold)
	sgd.Alpha = m.Alpha
	sgd.EnableLog = !m.scaffolded
	sgd.Cost = &logreg_loss[T]{
		Regularizer: m.create_regularizer(),
	}

	if err := sgd.Fit(ds); err != nil {
		return err
	}

	m.theta = sgd.GetParams()
	return nil
}

type logreg_loss[T constraints.Float] struct {
	Regularizer maths.BatchFunction[T]
}

func (l *logreg_loss[T]) On(params []T, ds *dataset.DataSet[T]) (T, error) {
	var r T
	if l.Regularizer != nil {
		lr, err := l.Regularizer.On(params, ds)
		if err != nil {
			return 0.0, err
		}
		r = lr
	}

	s, err := scaled_negative_log_likelihood(params, ds)
	if err != nil {
		return 0.0, err
	}

	return s + r, nil
}

func (l *logreg_loss[T]) Diff(j int, params []T, ds *dataset.DataSet[T]) (T, error) {
	var r T
	if l.Regularizer != nil {
		lr, err := l.Regularizer.Diff(j, params, ds)
		if err != nil {
			return 0.0, err
		}
		r = lr
	}

	var caught_err error
	dl := accumulator.Mean(iterable.Map(ds.Samples(), func(sample dataset.DataSample[T]) T {
		if caught_err != nil {
			return 0.0
		}

		y := sample.GetTarget()
		if y == nil {
			caught_err = fmt.Errorf("logreg_loss.Diff : No target found for row : %v", sample.GetRow())
			return 0.0
		}

		h, err := logreg_hypothesis_sample(params, sample)
		if err != nil {
			caught_err = err
			return 0.0
		}

		if j == 0 {
			return h - *y
		}

		var x T
		if j == 0 {
			x = 1.0
		} else {
			x = *sample.GetFeat(j - 1)
		}
		return x * (h - *y)
	}))

	if caught_err != nil {
		return 0.0, caught_err
	}

	return dl + r, nil
}

func (m *LogisticRegression[T]) create_regularizer() maths.BatchFunction[T] {
	switch m.Penalty {
	case regularization.Lasso:
		lasso := regularization.NewLasso[T](m.HyperParams.Lambda)
		return &lasso
	case regularization.Ridge:
		ridge := regularization.NewRidge[T](m.HyperParams.Lambda)
		return &ridge
	case regularization.ElasticNet:
		e := regularization.NewElasticnet[T](m.HyperParams.Lambda, m.HyperParams.Alpha)
		return &e
	}
	return nil
}

func (m *LogisticRegression[T]) Loss(ds *dataset.DataSet[T]) (T, error) {
	loss := logreg_loss[T]{
		Regularizer: m.create_regularizer(),
	}
	return loss.On(m.theta, ds)
}
