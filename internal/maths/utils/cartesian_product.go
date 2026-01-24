package utils

import (
	"maps"
	"slices"
)

func CartesianProductM[K comparable, V any](m map[K][]V) []map[K]V {
	keys := slices.Collect(maps.Keys(m))
	sets := make([][]V, len(m))
	for i, k := range keys {
		sets[i] = m[k]
	}

	c := CartesianProduct(sets)
	comb := make([]map[K]V, len(c))

	for i, set := range c {
		comb[i] = make(map[K]V)
		for j, v := range set {
			comb[i][keys[j]] = v
		}
	}

	return comb
}

func CartesianProduct[T any](sets [][]T) [][]T {
	if len(sets) == 0 {
		return nil
	}

	var res [][]T
	cur := make([]T, len(sets))

	var backtrack func(int)
	backtrack = func(i int) {
		if i == len(sets) {
			res = append(res, slices.Clone(cur))
			return
		}

		for _, v := range sets[i] {
			cur[i] = v
			backtrack(i + 1)
		}
	}

	backtrack(0)
	return res
}
