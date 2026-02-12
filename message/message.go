package message

type MessageType string

const (
	EXECUTE       MessageType = "execute"
	ABORT         MessageType = "abort"
	INFO          MessageType = "info"
	ERROR         MessageType = "error"
	FULFILLED     MessageType = "fulfilled"
	PAUSE         MessageType = "pause"
	RESUME        MessageType = "resume"
	EXEC_FINISHED MessageType = "exec_finished"
	STATS         MessageType = "stats"
)

type Message struct {
	Type MessageType `json:"type"`
	Data any         `json:"data"`
}
