package utils

import (
	"maps"
	"slices"
	"testing"
)

func TestCartesianProductM(t *testing.T) {
	comb := CartesianProductM(map[string][]string{
		"first":  {"a1", "a2"},
		"second": {"b1", "b2"},
	})
	expected := []map[string]string{
		{"first": "a1", "second": "b1"},
		{"first": "a1", "second": "b2"},
		{"first": "a2", "second": "b1"},
		{"first": "a2", "second": "b2"},
	}

	if !slices.EqualFunc(comb, expected, func(m1, m2 map[string]string) bool { return maps.Equal(m1, m2) }) {
		t.Errorf("Wrong combinations. Expected : %v. Got : %v", expected, comb)
	}
}

func TestCartesianProduct_OneVector(t *testing.T) {
	comb := CartesianProduct([][]string{{"a", "b"}})
	if len(comb) != 2 {
		t.Errorf("Expected 2 combinations, got : %v", len(comb))
		t.FailNow()
	}

	if !slices.Equal(comb[0], []string{"a"}) || !slices.Equal(comb[1], []string{"b"}) {
		t.Errorf("Wrong combinations. Expected [[a], [b]], Got : %v", comb)
	}
}

func TestCartesianProduct(t *testing.T) {
	sets := [][]string{
		{"a", "b"},
		{"f"},
	}

	comb := CartesianProduct(sets)
	if len(comb) != 2 {
		t.Errorf("There should be 2 combinations, got : %v", len(comb))
		t.FailNow()
	}

	if !slices.Equal(comb[0], []string{"a", "f"}) {
		t.Errorf("Wrong first combination. Expected [a, f] Got : %v", comb[0])
	}

	if !slices.Equal(comb[1], []string{"b", "f"}) {
		t.Errorf("Wrong first combination. Expected [b, f] Got : %v", comb[0])
	}
}
