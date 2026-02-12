package main

import (
	"encoding/json"
	"fmt"
	"time"
)

type Message struct {
	Type string `json:"type"`
	Data any    `json:"data"`
}

func main() {
	for i := range 200 {
		msg, _ := json.Marshal(Message{
			Type: "info",
			Data: i,
		})

		fmt.Println(string(msg))
		time.Sleep(time.Second)
	}
}
