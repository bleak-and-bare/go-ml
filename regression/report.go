package regression

import "github.com/bleak-and-bare/machine_learning/common/dataset"

type float interface {
	~float32 | ~float64
}

type RegressionReport[T float] struct {
	DataSet     *dataset.DataSet[T]
	Predictions []T
	SkippedRows int
	Error       float64
}
