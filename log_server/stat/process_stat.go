package stat

import "time"

type ProcessStat struct {
	Duration   time.Duration `json:"duration"`
	UserTime   time.Duration `json:"user_time"`
	SystemTime time.Duration `json:"system_time"`
	ExitStatus int           `json:"exit_status"`
}
