package optimization

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"slices"
	"sync"

	"github.com/bleak-and-bare/go-ml/machine_learning/common/dataset"
	"github.com/bleak-and-bare/go-ml/machine_learning/common/maths"
	"golang.org/x/exp/constraints"
)

// Stochastic Gradient Descent
type GradientDescent[T constraints.Float] struct {
	theta     []T
	BatchSize int
	Alpha     float32 // learning rate
	Threshold maths.Threshold
	Cost      maths.BatchFunction[T]
	EnableLog bool
}

// Stochastic Gradient Descent
type SGD[T constraints.Float] struct {
	GradientDescent[T]
}

func NewSGD[T constraints.Float](t maths.Threshold) GradientDescent[T] {
	return GradientDescent[T]{
		theta:     nil,
		BatchSize: 128,
		Alpha:     1e-4,
		Threshold: t,
		EnableLog: true,
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
	num_workers := min(runtime.GOMAXPROCS(0), len(g.theta))

	var wg sync.WaitGroup
	jobs := make(chan int, len(g.theta))

	for range num_workers {
		wg.Go(func() {
			for j := range jobs {
				g_j, _ := g.Cost.Diff(j, g.theta, ds)
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

	sample_size := int(ds.Size())
	n_theta := make([]T, len(g.theta))

	prev_cost, err := g.Cost.On(g.theta, ds)
	if err != nil {
		return err
	}

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
			ctx, cancel := context.WithCancel(context.Background())

			var wg sync.WaitGroup
			num_workers := min(runtime.GOMAXPROCS(0), len(g.theta))

			jobs := make(chan int)
			err_ch := make(chan error, 1)

			for range num_workers {
				wg.Go(func() {
					for {
						select {
						case <-ctx.Done():
							return
						case j, ok := <-jobs:
							if !ok {
								return
							}

							c, err := g.Cost.Diff(j, g.theta, batch)
							if err != nil {
								select {
								case err_ch <- err:
								default:
								}

								cancel()
								return
							}

							t := g.theta[j] - T(g.Alpha)*c
							if math.IsNaN(float64(t)) || math.IsInf(float64(t), 0) {
								select {
								case err_ch <- fmt.Errorf("Invalid parameter update: encountered NaN/Inf at %d", j):
								default:
								}

								cancel()
								return
							}

							n_theta[j] = t
						}
					}
				})
			}

			go func() {
				defer close(jobs)

				for i := range g.theta {
					select {
					case <-ctx.Done():
						return
					case jobs <- i:
					}
				}
			}()

			wg.Wait()
			cancel()

			if err := func() error {
				select {
				case err := <-err_ch:
					return err
				default:
					return nil
				}
			}(); err != nil {
				return err
			}

			copy(g.theta, n_theta)
		}

		if epoch >= g.Threshold.MinEphocs {
			grad_norm := g.gradient_norm(ds)
			if grad_norm <= T(g.Threshold.GradEps) {
				if g.EnableLog {
					fmt.Printf("Hitting gradient breakpoint. Total epochs : %d\n", epoch+1)
				}
				break
			}

			cost, err := g.Cost.On(g.theta, ds)
			if err != nil {
				return err
			}

			rel_cost := math.Abs(float64(cost-prev_cost)) / max(1, math.Abs(float64(prev_cost)))
			prev_cost = cost

			if rel_cost <= float64(g.Threshold.CostEps) {
				if g.EnableLog {
					fmt.Printf("Hitting cost breakpoint. Total epochs : %d\n", epoch+1)
				}
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
