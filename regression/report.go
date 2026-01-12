package regression

import (
	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"golang.org/x/exp/constraints"
)

type RegressionReport[T constraints.Float] struct {
	DataSet           *dataset.DataSet[T]
	Predictions       []T
	SkippedRows       int
	RootMeanSquareErr float64
	MeanAbsoluteErr   float64
	Score             float64
}
