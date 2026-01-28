package stat

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

type MemStat struct {
	RSS     int64 // kB
	PeakRSS int64 // kB
}

func parse_kb(line string) (int64, error) {
	fields := strings.Fields(line)
	if len(fields) > 0 {
		rss, err := strconv.ParseInt(fields[0], 10, 64)
		if err != nil {
			return 0, err
		}
		return rss, nil
	}

	return 0, fmt.Errorf("parse_kb : no value to parse from line")
}

func ReadMemStat(pid int) (MemStat, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
	if err != nil {
		return MemStat{}, err
	}

	var m MemStat
	rss_found, peak_rss_found := false, false

	for line := range strings.SplitSeq(string(data), "\n") {
		if rss_found && peak_rss_found {
			break
		}

		cols := strings.Split(line, ":")

		if len(cols) > 1 {
			fields := strings.Fields(cols[1])
			value, err := parse_kb(fields[0])

			if strings.EqualFold(cols[0], "VmRSS") {
				if err != nil {
					return MemStat{}, nil
				}

				m.RSS = value
				rss_found = true
			}

			if strings.EqualFold(cols[0], "VmHWM") {
				if err != nil {
					return MemStat{}, nil
				}

				m.PeakRSS = value
				peak_rss_found = true
			}
		}
	}

	return m, nil
}
