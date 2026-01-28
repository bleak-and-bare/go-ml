package stat

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const (
	UTIME = 13
	STIME = 14
)

type CPUStat struct {
	User   time.Duration
	System time.Duration
}

func ticks_to_duration(ticks int64, hz int64) time.Duration {
	seconds := float64(ticks) / float64(hz)
	return time.Duration(seconds * float64(time.Second))
}

func ReadCPUStat(pid int) (CPUStat, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return CPUStat{}, err
	}

	clk_tck, err := exec.Command("getconf", "CLK_TCK").Output()
	if err != nil {
		return CPUStat{}, err
	}

	hz, err := strconv.ParseInt(strings.TrimSpace(string(clk_tck)), 10, 64)
	if err != nil {
		return CPUStat{}, err
	}

	fields := strings.Fields(string(data))
	utime, _ := strconv.ParseInt(fields[UTIME], 10, 64)
	stime, _ := strconv.ParseInt(fields[STIME], 10, 64)

	return CPUStat{
		User:   ticks_to_duration(utime, hz),
		System: ticks_to_duration(stime, hz),
	}, nil
}
