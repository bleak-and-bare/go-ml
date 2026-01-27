package main

import (
	"fmt"
	"net/http"
	"os"

	"github.com/bleak-and-bare/go-ml/log_server/ws"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
	// ReadBufferSize:  1024, WriteBufferSize: 1024,
}

func ServeWS(hub *ws.Hub, cmd chan<- ws.Command, w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "serve_ws : failed to upgrade connection %v", err)
		return
	}

	client := ws.NewClient(hub, conn)

	go client.WritePump()
	go client.ReadPump(cmd)
}
