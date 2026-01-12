package processing

import (
	"iter"

	"github.com/bleak-and-bare/machine_learning/common/dataset"
	"github.com/bleak-and-bare/machine_learning/common/iterable"
	"github.com/bleak-and-bare/machine_learning/common/iterable/adapter"
	"github.com/bleak-and-bare/machine_learning/common/maths"
	"golang.org/x/exp/constraints"
)

type StandardScaler[T constraints.Float] struct {
	mean  T
	stdev T
}

func (s *StandardScaler[T]) Fit(it iter.Seq[T]) {
	s.mean = maths.Mean(it)
	s.stdev = maths.Stdev(it)
}

func (s *StandardScaler[T]) InverseTransform(pred []T) []T {
	real := make([]T, len(pred))
	for i := range pred {
		real[i] = s.stdev*pred[i] + s.mean
	}
	return real
}

func (s *StandardScaler[T]) Transform(it iter.Seq[*T]) {
	stdev := T(1.0)
	if s.stdev > 0.0 {
		stdev = s.stdev
	}

	for p := range it {
		if p != nil {
			*p = (*p - s.mean) / stdev
		}
	}
}

func (s *StandardScaler[T]) FitTransform(it iter.Seq[*T]) {
	s.Fit(adapter.PtrDerefAdapter(it))
	s.Transform(it)
}

func (s *StandardScaler[T]) FitTransformDataSet(ds *dataset.DataSet[T], col string) {
	ptr_it := iterable.Map(ds.Column(col), func(v dataset.DataCell) *T {
		if v == nil || !v.IsReal() {
			return nil
		}

		c, _ := v.(*dataset.RealDataCell[T])
		return &c.Value
	})

	s.FitTransform(ptr_it)
}

func (s *StandardScaler[T]) TransformDataSet(ds *dataset.DataSet[T], col string) {
	s.Transform(iterable.Map(ds.Column(col), func(v dataset.DataCell) *T {
		if v == nil || !v.IsReal() {
			return nil
		}

		c, _ := v.(*dataset.RealDataCell[T])
		return &c.Value
	}))
}
