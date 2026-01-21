package processing

import (
	"iter"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/iterable"
	"github.com/bleak-and-bare/machine_learning/internal/iterable/adapter"
	"github.com/bleak-and-bare/machine_learning/internal/maths/stat"
	"golang.org/x/exp/constraints"
)

type StandardScaler[T constraints.Float] struct {
	mean  T
	stdev T
}

func (s *StandardScaler[T]) Fit(it iter.Seq[T]) {
	s.mean = stat.Mean(it)
	s.stdev = stat.Stdev(it)
}

func (s *StandardScaler[T]) InverseTransform(sample iter.Seq[T]) []T {
	real := make([]T, 0, 1)
	for i := range sample {
		real = append(real, s.stdev*i+s.mean)
	}
	return real
}

func (s *StandardScaler[T]) Transform(it iter.Seq[*T]) {
	for p := range it {
		if p != nil {
			*p = s.TransformOne(*p)
		}
	}
}

func (s *StandardScaler[T]) TransformOne(v T) T {
	stdev := T(1.0)
	if s.stdev > 0.0 {
		stdev = s.stdev
	}
	return (v - s.mean) / stdev
}

func (s *StandardScaler[T]) FitTransform(it iter.Seq[*T]) {
	s.Fit(adapter.PtrDeref(it))
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
