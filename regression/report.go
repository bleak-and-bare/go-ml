package regression

import (
	"slices"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/maths/metrics"
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

func ComputeReport[T constraints.Float](predictions []T, targets []T) RegressionReport[T] {
	return RegressionReport[T]{
		Predictions:       predictions,
		Score:             metrics.R2Score(slices.Values(targets), slices.Values(predictions)),
		RootMeanSquareErr: metrics.RMSE(slices.Values(targets), slices.Values(predictions)),
		MeanAbsoluteErr:   metrics.MAE(slices.Values(targets), slices.Values(predictions)),
	}
}
