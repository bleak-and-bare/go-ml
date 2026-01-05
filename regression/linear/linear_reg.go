package regression

import (
	"errors"
	"fmt"
	"math/rand"
	"slices"

	"github.com/bleak-and-bare/machine_learning/common/dataset"
	"github.com/bleak-and-bare/machine_learning/common/maths"
)

type float interface {
	~float32 | ~float64
}

type LinearRegression[T float] struct {
	theta     []T     // parameter list
	Alpha     float32 // learning rate
	Epsilon   float32
	MaxEpochs uint32
}

func NewLinearReg[T float]() LinearRegression[T] {
	return LinearRegression[T]{
		Alpha:     1e-2,
		Epsilon:   1e-3,
		MaxEpochs: 1000,
	}
}

func (m *LinearRegression[T]) hypothesis(sample dataset.DataSample[T]) (T, error) {
	d, err := sample.DotProduct(m.theta[1:])
	if err != nil {
		return 0.0, err
	}

	return m.theta[0] + d, nil
}

func (m *LinearRegression[T]) next_bias(ds *dataset.DataSet[T], sample_size int) (T, error) {
	var sum T
	var caught_err error

	if !ds.ForEachSample(func(ds dataset.DataSample[T]) bool {
		y := ds.GetTarget()
		if y != nil {
			h, err := m.hypothesis(ds)
			if err != nil {
				caught_err = err
				return false
			}
			sum += h - *y
			return true
		}
		return false
	}) {
		return 0.0, caught_err
	}

	return m.theta[0] - T(m.Alpha/float32(sample_size))*sum, nil
}

// compute estimation of every parameters
func (m *LinearRegression[T]) gradient_descent(ds *dataset.DataSet[T]) error {
	feat_count := ds.FeatCount()
	if feat_count == 0 {
		return errors.New("LinearRegression.partial_diffj_err : dataset has no feature")
	}

	sample_size := int(ds.Size())
	if sample_size == 0 {
		return errors.New("LinearRegression.partial_diffj_err : dataset has no sample")
	}

	n_theta := slices.Clone(m.theta)
	for epoch := 0; epoch < int(m.MaxEpochs); epoch++ {
		for j := range m.theta {
			if j == 0 {
				theta, err := m.next_bias(ds, sample_size)
				if err != nil {
					return err
				}
				n_theta[0] = theta
			} else {
				var sum T
				var caught_err error

				if !ds.ForEachSample(func(ds dataset.DataSample[T]) bool {
					y := ds.GetTarget()
					x := ds.GetFeat(j - 1)

					if y != nil && x != nil {
						h, err := m.hypothesis(ds)
						if err != nil {
							caught_err = err
							return false
						}
						sum += *x * (h - *y)
					}

					return true
				}) {
					return caught_err
				}

				n_theta[j] = m.theta[j] - T(m.Alpha/float32(sample_size))*sum
			}
		}

		d, _ := maths.L2Dist(n_theta, m.theta)
		if d < T(m.Epsilon) {
			fmt.Printf("Total epochs : %d\n", epoch)
			return nil
		}

		for i := range m.theta {
			m.theta[i] = n_theta[i]
		}
	}

	return nil
}

func (m *LinearRegression[T]) initialize_parameters(feat_count int) {
	m.theta = make([]T, feat_count+1)

	for i := range m.theta {
		m.theta[i] = T(rand.Float32())
	}
}

func (m *LinearRegression[T]) PrintRegLineEq() {
	fmt.Printf("y = ")
	for i, theta := range m.theta {
		if i == 0 {
			fmt.Printf("%.3f", theta)
		} else {
			fmt.Printf(" + %.3f*x%d", theta, i)
		}
	}
	fmt.Println("")
}

func (m *LinearRegression[T]) Fit(ds *dataset.DataSet[T]) error {
	m.initialize_parameters(ds.FeatCount())
	if err := m.gradient_descent(ds); err != nil {
		return err
	}

	fmt.Printf("LinearRegression model fit\n")
	return nil
}

func (m *LinearRegression[T]) Predict(x []T) (T, error) {
	if len(m.theta) == 0 {
		return 0.0, errors.New("Using non-fit model")
	}

	if len(x) != len(m.theta)-1 {
		return 0.0, errors.New("Invalid vector provided")
	}

	sum := m.theta[0]
	for i := range x {
		sum += x[i] * m.theta[i+1]
	}

	return sum, nil
}
