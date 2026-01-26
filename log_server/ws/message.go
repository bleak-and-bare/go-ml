package ws

import "fmt"

type MessageType string

const (
	EXECUTE MessageType = "execute"
	ABORT   MessageType = "abort"
	INFO    MessageType = "info"
	ERROR   MessageType = "error"
)

type Message struct {
	Type MessageType `json:"type"`
	Data string      `json:"data"`
}

func NewErrMessage(msg string) Message {
	return Message{
		Type: ERROR,
		Data: msg,
	}
}

func (m *Message) Interpret() error {
	switch m.Type {
	case EXECUTE:
		fmt.Println(m.Data)
	}
	return nil
}
