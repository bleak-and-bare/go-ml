package stat

type ProcessStat struct {
	Duration   Millisecond `json:"duration"`
	UserTime   Millisecond `json:"user_time"`
	SystemTime Millisecond `json:"system_time"`
	ExitStatus int         `json:"exit_status"`
}
