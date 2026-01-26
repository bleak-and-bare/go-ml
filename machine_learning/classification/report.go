package classification

// https://en.wikipedia.org/wiki/Precision_and_recall

import (
	"fmt"
	"strconv"

	"github.com/bleak-and-bare/go-ml/machine_learning/common/misc"
)

type ClassMetrics struct {
	Precision float64 // prediction correctness
	Recall    float64 // coverage of actual positives
	F1        float64 // precision-recall balance
	Support   int     // number of true samples
}

type ClassificationReport struct {
	PerClass    map[int]ClassMetrics
	MacroAvg    ClassMetrics // unweighted class mean
	WeightedAvg ClassMetrics // support weighted-mean
}

type counts struct {
	TP, FP, FN int
}

func ComputeReport(trg, pred []int) ClassificationReport {
	counts := computeCounts(trg, pred)
	report := computeMetrics(counts)

	grid := misc.GridPrinter{
		Tab: "  ",
	}
	headers := []string{"class", "precision", "recall", "f1-score", "support"}
	for _, h := range headers {
		grid.Column(h)
	}

	for label, m := range report.PerClass {
		grid.NewRow()
		grid.Columns(
			strconv.Itoa(label),
			fmt.Sprintf("%.3f", m.Precision),

			fmt.Sprintf("%.3f", m.Recall),
			fmt.Sprintf("%.3f", m.F1),
			strconv.Itoa(m.Support),
		)
	}

	grid.NewEmptyRow()
	grid.NewRow()
	grid.Column("macro avg")
	grid.Columns(
		fmt.Sprintf("%.3f", report.MacroAvg.Precision),
		fmt.Sprintf("%.3f", report.MacroAvg.Recall),
		fmt.Sprintf("%.3f", report.MacroAvg.F1),
		strconv.Itoa(report.MacroAvg.Support),
	)

	grid.NewRow()
	grid.Column("weighted avg")
	grid.Columns(
		fmt.Sprintf("%.3f", report.WeightedAvg.Precision),
		fmt.Sprintf("%.3f", report.WeightedAvg.Recall),
		fmt.Sprintf("%.3f", report.WeightedAvg.F1),
		strconv.Itoa(report.WeightedAvg.Support),
	)
	grid.Print(true)

	return report
}

func computeCounts(trg, pred []int) map[int]*counts {
	c := make(map[int]*counts)

	for i := range trg {
		yt, yp := trg[i], pred[i]

		if _, ok := c[yt]; !ok {
			c[yt] = &counts{}
		}
		if _, ok := c[yp]; !ok {
			c[yp] = &counts{}
		}

		if yt == yp {
			c[yt].TP++
		} else {
			c[yp].FP++
			c[yt].FN++
		}
	}

	return c
}

func computeMetrics(cnt map[int]*counts) ClassificationReport {
	report := ClassificationReport{
		PerClass: make(map[int]ClassMetrics),
	}

	var (
		sumP, sumR, sumF float64
		totalSupport     int
	)

	for label, c := range cnt {
		tp, fp, fn := float64(c.TP), float64(c.FP), float64(c.FN)
		support := c.TP + c.FN

		precision := 0.0
		recall := 0.0
		f1 := 0.0

		if tp+fp > 0 {
			precision = tp / (tp + fp)
		}
		if tp+fn > 0 {
			recall = tp / (tp + fn)
		}
		if precision+recall > 0 {
			f1 = 2 * precision * recall / (precision + recall)
		}

		report.PerClass[label] = ClassMetrics{
			Precision: precision,
			Recall:    recall,
			F1:        f1,
			Support:   support,
		}

		sumP += precision
		sumR += recall
		sumF += f1
		totalSupport += support

		report.WeightedAvg.Precision += precision * float64(support)
		report.WeightedAvg.Recall += recall * float64(support)
		report.WeightedAvg.F1 += f1 * float64(support)
	}

	nClasses := float64(len(report.PerClass))

	report.MacroAvg = ClassMetrics{
		Precision: sumP / nClasses,
		Recall:    sumR / nClasses,
		F1:        sumF / nClasses,
		Support:   totalSupport,
	}

	if totalSupport > 0 {
		report.WeightedAvg.Precision /= float64(totalSupport)
		report.WeightedAvg.Recall /= float64(totalSupport)
		report.WeightedAvg.F1 /= float64(totalSupport)
		report.WeightedAvg.Support = totalSupport
	}

	return report
}
