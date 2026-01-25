package selector

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"slices"
	"time"

	"github.com/bleak-and-bare/machine_learning/internal/dataset"
	"github.com/bleak-and-bare/machine_learning/internal/iterable/accumulator"
	"github.com/bleak-and-bare/machine_learning/internal/maths/utils"
	"golang.org/x/exp/constraints"
)

// Choose hyper parameter based on estimator using cross validation
type GridSearch[T constraints.Float] struct {
	param_choices map[string][]float64
	model_factory ModelFactory[T]
	Fold          int // number of cross-validation
	best_params   map[string]float64
}

func NewGridSearch[T constraints.Float](param_choices map[string][]float64, model_factory ModelFactory[T]) GridSearch[T] {
	return GridSearch[T]{
		model_factory: model_factory,
		param_choices: param_choices,
		best_params:   nil,
		Fold:          5,
	}
}

func (g *GridSearch[T]) BestParams() map[string]float64 {
	return g.best_params
}

type grid_search_job[T constraints.Float] struct {
	params map[string]float64  // one combination of parameters
	ds     *dataset.DataSet[T] // clone of source dataset
}

type grid_search_result[T constraints.Float] struct {
	params map[string]float64
	loss   T
	err    error
}

func (g *GridSearch[T]) grid_search_worker(ctx context.Context, jobs <-chan grid_search_job[T], results chan<- grid_search_result[T]) {
	for {
		select {
		case <-ctx.Done():
			return
		case job, ok := <-jobs:
			if !ok {
				return
			}

			i := 0
			losses := make([]T, g.Fold)

			for r := range job.ds.HoldOutSplit(g.Fold) {
				if r.Err != nil {
					select {
					case <-ctx.Done():
						return
					case results <- grid_search_result[T]{err: r.Err}:
					}
					return
				}

				fold := r.Value
				m := g.model_factory(job.params)

				if err := m.Fit(fold.Train); err != nil {
					select {
					case <-ctx.Done():
						return
					case results <- grid_search_result[T]{err: err}:
					}
					return
				}

				loss, err := m.Loss(fold.Test)
				if err != nil {
					select {
					case <-ctx.Done():
						return
					case results <- grid_search_result[T]{err: err}:
					}
					return
				}

				losses[i] = loss
				i++
			}

			select {
			case <-ctx.Done():
				return
			case results <- grid_search_result[T]{
				loss:   T(accumulator.Mean(slices.Values(losses))),
				params: job.params,
			}:
			}
		}
	}
}

func (g *GridSearch[T]) Fit(ds *dataset.DataSet[T]) error {
	start := time.Now()
	defer func() {
		elapsed := time.Since(start)
		fmt.Printf("GridSearch cross-validation took %v\n", elapsed)
	}()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	combinations := utils.CartesianProductM(g.param_choices)
	results := make(chan grid_search_result[T])
	jobs := make(chan grid_search_job[T])
	num_workers := min(len(combinations), runtime.GOMAXPROCS(0))

	for range num_workers {
		go g.grid_search_worker(ctx, jobs, results)
	}

	go func() {
		defer close(jobs)
		for _, comb := range combinations {
			ds_copy := ds.Copy()
			select {
			case <-ctx.Done():
				return
			case jobs <- grid_search_job[T]{
				params: comb,
				ds:     &ds_copy,
			}:
			}
		}
	}()

	best_loss := math.Inf(1)
	var best_params map[string]float64

	done := 0
	fmt.Printf("\nGridSearch progress : %d/%d (%d/%d%%)", 0, len(combinations), 0, 100)
	for range combinations {
		r := <-results
		done++

		if r.err != nil {
			return r.err
		}

		fmt.Printf("\r\033[KGridSearch progress : %d/%d (%d/%d%%)", done,
			len(combinations),
			int(math.Round(100.0*float64(done)/float64(len(combinations)))), 100)

		if r.loss < T(best_loss) {
			best_loss = float64(r.loss)
			best_params = r.params
		}
	}
	fmt.Println()

	g.best_params = best_params

	return nil
}
