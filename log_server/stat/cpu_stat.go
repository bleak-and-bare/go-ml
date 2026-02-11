package stat

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

const (
	PROC_UTIME = 13
	PROC_STIME = 14
)

var JIFFIES *int64

type cpu_stat struct {
	User   Millisecond `json:"user"`
	System Millisecond `json:"system"`
}

type CPUStat struct {
	Process cpu_stat    `json:"process"`
	Global  Millisecond `json:"global"`
}

func ticks_to_duration(ticks int64, hz float32) Millisecond {
	seconds := float32(ticks) * hz
	return Millisecond(seconds * 1000)
}

func get_jiffies() error {
	clk_tck, err := exec.Command("getconf", "CLK_TCK").Output()
	if err != nil {
		return err
	}

	jiffies, err := strconv.ParseInt(strings.TrimSpace(string(clk_tck)), 10, 64)
	if err != nil {
		return err
	}

	JIFFIES = &jiffies
	return nil
}

func ReadCPUStat(pid int) (CPUStat, error) {
	if JIFFIES == nil {
		if err := get_jiffies(); err != nil {
			return CPUStat{}, err
		}
	}

	proc, err := read_proc_stat(pid)
	if err != nil {
		return CPUStat{}, err
	}

	glob, err := read_global_stat()
	if err != nil {
		return CPUStat{}, err
	}

	return CPUStat{
		Process: proc,
		Global:  glob,
	}, nil
}

func read_global_stat() (Millisecond, error) {
	stats, err := os.Open("/proc/stat")
	if err != nil {
		return 0, err
	}
	defer stats.Close()

	scanner := bufio.NewScanner(stats)
	if scanner.Scan() {
		fields := strings.Fields(string(scanner.Text()))
		var usage int64

		for _, field := range fields[1:] {
			v, err := strconv.ParseInt(field, 10, 64)
			if err != nil {
				continue
			}

			usage += v
		}

		hz := 1 / float32(*JIFFIES)
		return ticks_to_duration(usage, hz), nil
	}

	return 0, fmt.Errorf("ReadCPUStat: failed to read /proc/stat")
}

func read_proc_stat(pid int) (cpu_stat, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return cpu_stat{}, err
	}

	fields := strings.Fields(string(data))
	utime, _ := strconv.ParseInt(fields[PROC_UTIME], 10, 64)
	stime, _ := strconv.ParseInt(fields[PROC_STIME], 10, 64)
	hz := 1 / float32(*JIFFIES)

	return cpu_stat{
		User:   ticks_to_duration(utime, hz),
		System: ticks_to_duration(stime, hz),
	}, nil
}
