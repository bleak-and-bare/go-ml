package linear

import (
	"fmt"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/iterable"
	"github.com/bleak-and-bare/machine_learning/internal/iterable/accumulator"
	"github.com/bleak-and-bare/machine_learning/internal/maths"
	"github.com/bleak-and-bare/machine_learning/internal/maths/optimization"
	"github.com/bleak-and-bare/machine_learning/internal/maths/regularization"
	"github.com/bleak-and-bare/machine_learning/internal/selector"
	"golang.org/x/exp/constraints"
)

func (m *LogisticRegression[T]) Fit(ds *dataset.DataSet[T]) error {
	if m.hyper_params != nil || m.Penalty == regularization.None {
		sgd := optimization.NewSGD[T](m.Threshold)
		sgd.Alpha = m.Alpha
		sgd.EnableLog = !m.scaffolded
		sgd.Cost = &logreg_loss[T]{
			Regularizator: m.create_regularizator(),
		}

		if err := sgd.Fit(ds); err != nil {
			return err
		}

		m.theta = sgd.GetParams()
		return nil
	}

	param_choices := regularization.ElasticnetDefGrid()
	switch m.Penalty {
	case regularization.Lasso:
		param_choices.Alpha = []float64{1.0}
		break
	case regularization.Ridge:
		param_choices.Alpha = []float64{0.0}
		break
	}

	gs := selector.NewGridSearch([][]float64{param_choices.Alpha, param_choices.Lambda}, logreg_factory(*m))
	if err := gs.Fit(ds); err != nil {
		return err
	}

	hyper_params := gs.BestParams()
	m.hyper_params = &struct {
		alpha  float64
		lambda float64
	}{
		alpha:  hyper_params[0],
		lambda: hyper_params[1],
	}

	return m.Fit(ds)
}

type logreg_loss[T constraints.Float] struct {
	Regularizator maths.BatchFunction[T]
}

func (l *logreg_loss[T]) On(params []T, ds *dataset.DataSet[T]) (T, error) {
	var r T
	if l.Regularizator != nil {
		lr, err := l.Regularizator.On(params, ds)
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
	if l.Regularizator != nil {
		lr, err := l.Regularizator.Diff(j, params, ds)
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

func (m *LogisticRegression[T]) create_regularizator() maths.BatchFunction[T] {
	switch m.Penalty {
	case regularization.Lasso:
		lasso := regularization.NewLasso[T](m.hyper_params.lambda)
		return &lasso
	case regularization.Ridge:
		ridge := regularization.NewRidge[T](m.hyper_params.lambda)
		return &ridge
	case regularization.ElasticNet:
		e := regularization.Elasticnet[T]{
			Alpha:  m.hyper_params.alpha,
			Lambda: m.hyper_params.lambda,
		}
		return &e
	}
	return nil
}

func logreg_factory[T constraints.Float](blueprint LogisticRegression[T]) selector.ModelFactory[T] {
	return func(hyper_params []float64) selector.Model[T] {
		m := blueprint
		m.theta = nil
		m.scaffolded = true
		m.hyper_params = &struct {
			alpha  float64
			lambda float64
		}{
			alpha:  hyper_params[0],
			lambda: hyper_params[1],
		}
		return &m
	}
}

func (m *LogisticRegression[T]) Loss(ds *dataset.DataSet[T]) (T, error) {
	loss := logreg_loss[T]{
		Regularizator: m.create_regularizator(),
	}
	return loss.On(m.theta, ds)
}
