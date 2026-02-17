package message

type ProcessInfos struct {
	User   int64 `json:"user"`
	System int64 `json:"system"`
}

type ProcessStat struct {
	Duration   int64 `json:"duration"`
	UserTime   int64 `json:"user_time"`
	SystemTime int64 `json:"system_time"`
	ExitStatus int   `json:"exit_status"`
}

type StatStruct struct {
	Process ProcessInfos `json:"process"`
	Global  int64        `json:"global"`
	RSS     int64        `json:"rss"`
	PeakRSS int64        `json:"peak_rss"`
	Delta   int64        `json:"delta"`
}
