package message

import (
	"encoding/json"
	"fmt"
)

func print_msg(msg Message) {
	bytes, _ := json.Marshal(msg)
	fmt.Println(string(bytes))
}

func Info(msg string) {
	print_msg(NewInfo(msg))
}

func Errorf(with_stack_frame bool, format string, values ...any) {
	Error(fmt.Sprintf(format, values...), with_stack_frame)
}

func Error(msg string, with_stack_frame bool) {
	print_msg(NewError(msg, with_stack_frame))
}

func Table(caption string, head []string, body [][]string) {
	print_msg(Message{
		Type: TABLE,
		Data: TableStruct{
			Caption: caption,
			Head:    head,
			Body:    body,
		},
	})
}

func Progress(label string, id string, value float32) {
	print_msg(NewProgress(label, id, value))
}
