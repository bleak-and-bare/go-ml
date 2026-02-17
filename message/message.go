package message

import "runtime"

type MessageType string

const (
	EXECUTE       MessageType = "execute"
	ABORT         MessageType = "abort"
	INFO          MessageType = "info"
	PROGRESS      MessageType = "progress"
	TABLE         MessageType = "table"
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

func NewInfo(msg string) Message {
	return Message{
		Type: INFO,
		Data: msg,
	}
}

func NewError(msg string, with_stack_frame bool) Message {
	if !with_stack_frame {
		return Message{
			Type: ERROR,
			Data: ErrorStruct{
				Error: msg,
			},
		}
	}

	buf := make([]byte, 1064)
	n := runtime.Stack(buf, false)

	return Message{
		Type: ERROR,
		Data: ErrorStruct{
			Error:      msg,
			StackFrame: string(buf[:n]),
		},
	}
}

func NewAbort() Message {
	return Message{Type: ABORT}
}

func NewFulfilled(req_type MessageType) Message {
	return Message{
		Type: FULFILLED,
		Data: req_type,
	}
}

func NewStats(stats StatStruct) Message {
	return Message{
		Type: STATS,
		Data: stats,
	}
}

func NewExecFinished(stats ProcessStat) Message {
	return Message{
		Type: EXEC_FINISHED,
		Data: stats,
	}
}
