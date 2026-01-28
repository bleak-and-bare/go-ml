package stat

import "time"

type Stat struct {
	UserCPU   time.Duration `json:"user_cpu"`
	SystemCPU time.Duration `json:"system_cpu"`
	RSS       int64         `json:"rss"`      // kB
	PeakRSS   int64         `json:"peak_rss"` // kB
	Delta     time.Duration `json:"delta"`
}

func NewStat(cpu CPUStat, mem MemStat, dt time.Duration) Stat {
	return Stat{
		UserCPU:   cpu.User,
		SystemCPU: cpu.System,
		RSS:       mem.RSS,
		PeakRSS:   mem.PeakRSS,
		Delta:     dt,
	}
}
