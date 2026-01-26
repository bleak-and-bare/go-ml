package maths

import (
	"github.com/bleak-and-bare/go-ml/machine_learning/common/dataset"
	"golang.org/x/exp/constraints"
)

type SampleFunction[T constraints.Float] interface {
	// Compute function value on given data sample
	// parameters :
	// - params : commonly called theta. With n+1 length where theta0 is the bias
	// - sample : one sample used for estimation
	On(params []T, sample *dataset.DataSample[T]) (T, error)

	// Compute function value on given data sample
	// parameters :
	// - j : index of theta variable to apply derivative. j == 0 means derivating regarding the bias theta0 and pulling the first feature with sample.GetFeat(0) might not what you want in that regard
	// - params : commonly called theta
	// - sample : one sample used for estimation
	Diff(j int, params []T, sample *dataset.DataSample[T]) (T, error)
}

type BatchFunction[T constraints.Float] interface {
	// Compute function value on given dataset
	// parameters :
	// - params : commonly called theta. With n+1 length where theta0 is the bias
	// - ds : dataset
	On(params []T, ds *dataset.DataSet[T]) (T, error)

	// Compute function value on given dataset
	// parameters :
	// - j : index of theta variable to apply derivative. j == 0 means derivating regarding the bias theta0 and pulling the first feature with sample.GetFeat(0) might not what you want in that regard
	// - params : commonly called theta
	// - ds : dataset
	Diff(j int, params []T, ds *dataset.DataSet[T]) (T, error)
}
