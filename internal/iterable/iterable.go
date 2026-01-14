package iterable

import (
	"iter"
)

// skip n-first iterations
func Skip[T any](it iter.Seq[T], n int) iter.Seq[T] {
	return func(yield func(T) bool) {
		for v := range it {
			if n > 0 {
				n--
				continue
			}

			if !yield(v) {
				return
			}
		}
	}
}

func Map[T any, V any](it iter.Seq[T], m func(T) V) iter.Seq[V] {
	return func(yield func(V) bool) {
		for t := range it {
			if !yield(m(t)) {
				return
			}
		}
	}
}

func Filter[T any](it iter.Seq[T], f func(T) bool) iter.Seq[T] {
	return func(yield func(T) bool) {
		for v := range it {
			if f(v) {
				if !yield(v) {
					return
				}
			}
		}
	}
}
