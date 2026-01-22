package utils

import "slices"

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
