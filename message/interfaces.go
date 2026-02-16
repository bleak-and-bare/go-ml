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
	print_msg(Message{
		Type: INFO,
		Data: msg,
	})
}

func Error(msg string) {
	print_msg(Message{
		Type: ERROR,
		Data: msg,
	})
}
