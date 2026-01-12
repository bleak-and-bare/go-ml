package optimization

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"slices"
	"sync"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/maths"
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
		Alpha:     1e-4,
		Epsilon:   1e-5,
		MaxEpochs: 10_1000,
	}
}

func (g *GradientDescent[T]) GetParams() []T {
	return g.theta
}

func (g *GradientDescent[T]) initialize_parameters(ds *dataset.DataSet[T]) {
	g.theta = make([]T, ds.FeatCount()+1)
	g.theta[0] = ds.TargetMean()
}

func (g *GradientDescent[T]) process(ds *dataset.DataSet[T]) error {
	if g.CostPartialDiff == nil {
		return errors.New("No partial derivative function supplied")
	}

	sample_size := int(ds.Size())
	n_theta := make([]T, len(g.theta))

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
			var wg sync.WaitGroup
			num_workers := min(runtime.NumCPU(), len(g.theta))

			jobs := make(chan int, len(g.theta))
			err_ch := make(chan error, len(g.theta))

			for range num_workers {
				wg.Go(func() {
					for j := range jobs {
						c, err := g.CostPartialDiff(j, g.theta, batch)
						if err != nil {
							err_ch <- err
							return
						}

						n_theta[j] = g.theta[j] - T(g.Alpha)*c
						if math.IsNaN(float64(n_theta[j])) || math.IsInf(float64(n_theta[j]), 0) {
							panic("WTF ?")
						}
					}
				})
			}

			for i := range g.theta {
				jobs <- i
			}
			close(jobs)

			wg.Wait()
			close(err_ch)

			for err := range err_ch {
				return err
			}

			stop_here := false
			dist := maths.L2Dist(slices.Values(n_theta), slices.Values(g.theta))
			norm_theta := maths.L2Norm(slices.Values(g.theta))
			coef := dist / T(norm_theta+1e-8)

			// coefficient of determination: relative convergence criterion
			if coef < T(g.Epsilon) {
				fmt.Printf("Total epochs : %d\n", epoch+1)
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
	g.initialize_parameters(ds)
	if err := g.process(ds); err != nil {
		return err
	}
	return nil
}
