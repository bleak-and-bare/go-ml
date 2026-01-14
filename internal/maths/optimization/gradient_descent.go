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
	Threshold       maths.Threshold
	Cost            func(theta []T, ds *dataset.DataSet[T]) T
	CostPartialDiff func(j int, theta []T, ds *dataset.DataSet[T]) (T, error)
}

// Stochastic Gradient Descent
type SGD[T constraints.Float] struct {
	GradientDescent[T]
}

// Initialize SGD with MSE as cost function
func NewSGD[T constraints.Float](t maths.Threshold) GradientDescent[T] {
	return GradientDescent[T]{
		theta:     nil,
		BatchSize: 128,
		Alpha:     1e-4,
		Threshold: t,
	}
}

func (g *GradientDescent[T]) GetParams() []T {
	return g.theta
}

func (g *GradientDescent[T]) initialize_parameters(ds *dataset.DataSet[T]) {
	g.theta = make([]T, ds.FeatCount()+1)
	g.theta[0] = ds.TargetMean()
}

func (g *GradientDescent[T]) gradient_norm(ds *dataset.DataSet[T]) T {
	grad := make([]T, len(g.theta))
	num_workers := min(runtime.NumCPU(), len(g.theta))

	var wg sync.WaitGroup
	jobs := make(chan int, len(g.theta))

	for range num_workers {
		wg.Go(func() {
			for j := range jobs {
				g_j, _ := g.CostPartialDiff(j, g.theta, ds)
				grad[j] = g_j
			}
		})
	}

	for j := range g.theta {
		jobs <- j
	}
	close(jobs)

	wg.Wait()
	return maths.L2Norm(slices.Values(grad))
}

func (g *GradientDescent[T]) process(ds *dataset.DataSet[T]) error {
	if g.Cost == nil {
		return errors.New("No cost function supplied")
	}

	if g.CostPartialDiff == nil {
		return errors.New("No partial derivative function supplied")
	}

	sample_size := int(ds.Size())
	prev_cost := g.Cost(g.theta, ds)
	n_theta := make([]T, len(g.theta))

	for epoch := 0; epoch < int(g.Threshold.MaxEpochs); epoch++ {
		var batches []*dataset.DataSet[T]

		if sample_size > g.BatchSize {
			ds.Shuffle()
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

			copy(g.theta, n_theta)
		}

		if epoch >= g.Threshold.MinEphocs {
			grad_norm := g.gradient_norm(ds)
			if grad_norm <= T(g.Threshold.GradEps) {
				fmt.Printf("Hitting gradient breakpoint. Total epochs : %d\n", epoch+1)
				break
			}

			cost := g.Cost(g.theta, ds)
			rel_cost := math.Abs(float64(cost-prev_cost)) / max(1, math.Abs(float64(prev_cost)))
			prev_cost = cost

			if rel_cost <= float64(g.Threshold.CostEps) {
				fmt.Printf("Hitting cost breakpoint. Total epochs : %d\n", epoch+1)
				break
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
