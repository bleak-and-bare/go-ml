package iterable

import (
	"iter"
)

func Append[T any](it iter.Seq[T], values ...T) iter.Seq[T] {
	return func(yield func(T) bool) {
		for v := range it {
			if !yield(v) {
				return
			}
		}

		for _, v := range values {
			if !yield(v) {
				return
			}
		}
	}
}

func Prepend[T any](it iter.Seq[T], values ...T) iter.Seq[T] {
	return func(yield func(T) bool) {
		for _, v := range values {
			if !yield(v) {
				return
			}
		}

		for v := range it {
			if !yield(v) {
				return
			}
		}
	}
}

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

func Pointers[T any](it []T) iter.Seq[*T] {
	return func(yield func(*T) bool) {
		for i := range it {
			if !yield(&it[i]) {
				return
			}
		}
	}
}

// Do remember that this is a O(n) operation
func Count[T any](it iter.Seq[T]) int {
	count := 0
	for range it {
		count++
	}
	return count
}
