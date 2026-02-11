package stat

import "time"

type Stat struct {
	CPUStat
	RSS     int64       `json:"rss"`      // kB
	PeakRSS int64       `json:"peak_rss"` // kB
	Delta   Millisecond `json:"delta"`
}

func NewStat(cpu CPUStat, mem MemStat, dt time.Duration) Stat {
	return Stat{
		CPUStat: cpu,
		RSS:     mem.RSS,
		PeakRSS: mem.PeakRSS,
		Delta:   Millisecond(dt.Milliseconds()),
	}
}
