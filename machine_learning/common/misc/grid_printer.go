package misc

import (
	"fmt"
	"strings"
)

type GridPrinter struct {
	max_col_lengths []int
	grid            [][]string
	Tab             string
}

func (g *GridPrinter) Clear() {
	g.grid = nil
	g.max_col_lengths = nil
}

func (g *GridPrinter) Print(show_header bool) {
	fmt.Printf("%v\n", g.String(show_header))
}

func (g *GridPrinter) String(show_header bool) string {
	var sb strings.Builder
	line := ""

	for _, row := range g.grid {
		for j, col := range row {
			blank := strings.Repeat(" ", g.max_col_lengths[j]-len(col))
			fmt.Fprintf(&sb, "| %v%v%v ", col, blank, g.Tab)
		}
		sb.WriteRune('|')

		if len(line) == 0 {
			line = strings.Repeat("─", len(sb.String())-2)
			if show_header {
				sb.WriteString("\n├" + line + "┤")
			}
		}
		sb.WriteRune('\n')
	}

	return fmt.Sprintf("%v\n%v%v",
		"┌"+line+"┐",
		sb.String(),
		"└"+line+"┘",
	)
}

func (g *GridPrinter) NewRow() {
	if len(g.grid) > 0 {
		g.grid = append(g.grid, make([]string, 0, len(g.grid[len(g.grid)-1])))
	} else {
		g.grid = append(g.grid, make([]string, 0))
	}
}

func (g *GridPrinter) NewEmptyRow() {
	cols := 0
	if len(g.grid) > 0 {
		cols = len(g.grid[len(g.grid)-1])
	}

	g.NewRow()
	for range cols {
		g.Column("")
	}
}

func (g *GridPrinter) Columns(cells ...string) {
	for _, c := range cells {
		g.Column(c)
	}
}

func (g *GridPrinter) Column(cell string) {
	if len(g.grid) == 0 {
		g.NewRow()
	}

	g.grid[len(g.grid)-1] = append(g.grid[len(g.grid)-1], cell)
	if len(g.grid[len(g.grid)-1]) <= len(g.max_col_lengths) {
		cur_cell := len(g.grid[len(g.grid)-1]) - 1
		g.max_col_lengths[cur_cell] = max(g.max_col_lengths[cur_cell], len(cell))
	} else {
		g.max_col_lengths = append(g.max_col_lengths, len(cell))
	}
}
