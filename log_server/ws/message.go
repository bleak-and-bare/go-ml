package ws

type MessageType string

const (
	EXECUTE MessageType = "execute"
	ABORT   MessageType = "abort"
	INFO    MessageType = "info"
	ERROR   MessageType = "error"
	PAUSE   MessageType = "pause"
	RESUME  MessageType = "resume"
)

type Message struct {
	Type MessageType `json:"type"`
	Data any         `json:"data"`
}
