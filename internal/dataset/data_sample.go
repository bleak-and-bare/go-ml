package dataset

import (
	"fmt"

	"golang.org/x/exp/constraints"
)

type DataSample[T constraints.Float] struct {
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

func (s *DataSample[T]) GetSampleTest() ([]T, error) {
	sample := make([]T, s.owner.FeatCount())
	for i := range sample {
		feat := s.owner.GetFeat(s.row, i)
		if feat == nil {
			return nil, fmt.Errorf("DataSample.GetSampleTest : failed to get feature <%d, %d>", s.row, i)
		}
		sample[i] = *feat
	}

	return sample, nil
}

// placeholder : default value for empty cells
func (s *DataSample[T]) GetSampleTestNoErr(placeholder T) []T {
	sample := make([]T, s.owner.FeatCount())
	for i := range sample {
		feat := s.owner.GetFeat(s.row, i)
		if feat == nil {
			sample[i] = placeholder
		} else {
			sample[i] = *feat
		}
	}

	return sample
}

func (s *DataSample[T]) GetFeat(i int) *T {
	return s.owner.GetFeat(s.row, i)
}

func (s *DataSample[T]) GetRow() int {
	return s.row
}

func (s *DataSample[T]) GetTarget() *T {
	trg := s.owner.at(s.row, int(s.owner.trg_col_idx))
	if trg.IsReal() {
		t, _ := trg.(*RealDataCell[T])
		return &t.Value
	}

	return nil
}
