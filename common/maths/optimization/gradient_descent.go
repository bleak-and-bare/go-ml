package optimization

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"slices"

	"github.com/bleak-and-bare/machine_learning/common/dataset"
	"github.com/bleak-and-bare/machine_learning/common/maths"
	"golang.org/x/exp/constraints"
)

// Stochastic Gradient Descent
type GradientDescent[T constraints.Float] struct {
	theta           []T
	BatchSize       int
	Alpha           float32 // learning rate
	Epsilon         float32
	MaxEpochs       uint32
	CostPartialDiff func(j int, theta []T, ds *dataset.DataSet[T]) (T, error)
}

// Stochastic Gradient Descent
type SGD[T constraints.Float] struct {
	GradientDescent[T]
}

// Initialize SGD with MSE as cost function
func NewSGD[T constraints.Float]() GradientDescent[T] {
	return GradientDescent[T]{
		theta:     nil,
		BatchSize: 128,
		Alpha:     1e-3,
		MaxEpochs: 10_1000,
	}
}

func (g *GradientDescent[T]) GetParams() []T {
	return g.theta
}

func (g *GradientDescent[T]) initialize_parameters(feat_count int) {
	g.theta = make([]T, feat_count+1)

	for i := range g.theta {
		g.theta[i] = T(rand.Float32())
	}
}

func (g *GradientDescent[T]) process(ds *dataset.DataSet[T]) error {
	if g.CostPartialDiff == nil {
		return errors.New("No partial derivative function supplied")
	}

	sample_size := int(ds.Size())
	n_theta := slices.Clone(g.theta)

	for epoch := 0; epoch < int(g.MaxEpochs); epoch++ {
		ds.Shuffle()
		var batches []*dataset.DataSet[T]

		if sample_size > g.BatchSize {
			for batch_i := 0; batch_i < sample_size; batch_i += g.BatchSize {
				batch, err := ds.Extract(float32(batch_i)/float32(sample_size), float32(batch_i+g.BatchSize)/float32(sample_size))
				if err != nil {
					return err
				}
				batches = append(batches, batch)
			}
		} else {
			batches = append(batches, ds)
		}

		for _, batch := range batches {
			for j := range g.theta {
				c, err := g.CostPartialDiff(j, g.theta, batch)
				if err != nil {
					return err
				}

				n_theta[j] = g.theta[j] - T(g.Alpha)*c
			}

			stop_here := false
			d, _ := maths.L2Dist(n_theta, g.theta)
			if d < T(g.Epsilon) {
				fmt.Printf("Total epochs : %d\n", epoch)
				stop_here = true
			}

			for i := range g.theta {
				g.theta[i] = n_theta[i]
			}

			if stop_here {
				return nil
			}
		}
	}

	return nil
}

func (g *GradientDescent[T]) Fit(ds *dataset.DataSet[T]) error {
	g.initialize_parameters(ds.FeatCount())
	if err := g.process(ds); err != nil {
		return err
	}
	return nil
}
