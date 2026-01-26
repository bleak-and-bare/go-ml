package ws

import "time"

const (
	MAX_MSG_SIZE = 512 * 512
	PONG_WAIT    = 60 * time.Second
	PING_PERIOD  = (PONG_WAIT * 9) / 10 // have to be smaller than PONG_WAIT
	WRITE_WAIT   = 10 * time.Second
)
