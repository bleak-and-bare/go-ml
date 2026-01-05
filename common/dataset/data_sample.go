package dataset

import (
	"fmt"
)

type DataSample[T float] struct {
	owner *DataSet[T]
	row   int
	// curr_feat int
}

func (s *DataSample[T]) DotProduct(v []T) (T, error) {
	var sum T
	for i := range v {
		f := s.owner.GetFeat(s.row, i)
		if f == nil {
			return 0.0, fmt.Errorf("DataSample.DotProduct : dataset has empty cell <row: %d, feat: %d>", s.row, i)
		}
		sum += *f * v[i]
	}

	return sum, nil
}

func (s *DataSample[T]) GetSampleTest(placeholder T) []T {
	sample := make([]T, s.owner.FeatCount())
	for i := range sample {
		sample[i] = *s.owner.GetFeat(s.row, i)
	}

	return sample
}

func (s *DataSample[T]) GetFeat(i int) *T {
	return s.owner.GetFeat(s.row, i)
}

func (s *DataSample[T]) GetTarget() *T {
	return s.owner.datas[s.row][s.owner.trg_col_idx]
}
