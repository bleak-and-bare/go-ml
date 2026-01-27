package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/bleak-and-bare/go-ml/log_server/command"
	"github.com/bleak-and-bare/go-ml/log_server/ws"
)

func main() {
	port := "8000"
	if p, ok := os.LookupEnv("SERVER_PORT"); ok {
		port = p
	}

	hub := ws.NewHub()
	go hub.Run()

	c := command.NewCmdController()
	go c.Run()

	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		ServeWS(hub, c.Commands(), w, r)
	})

	fmt.Printf("Listening on port %v\n", port)
	err := http.ListenAndServe(":"+port, nil)
	if err != nil {
		log.Fatal("http.ListenAndServe : ", err)
	}
}
