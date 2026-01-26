package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
}

func serve_ws(hub *Hub, w http.ResponseWriter, r *http.Request) {
	upgrader.CheckOrigin = func(r *http.Request) bool {
		return true
	}
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "serve_ws : failed to upgrade connection %v", err)
		return
	}

	client := NewClient(hub, conn)

	go client.WritePump()
	go client.ReadPump()
}

func main() {
	port := "8000"
	if p, ok := os.LookupEnv("PORT"); ok {
		port = p
	}

	hub := NewHub()
	go hub.Run()

	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		serve_ws(hub, w, r)
	})

	fmt.Printf("Listening on port %v\n", port)
	err := http.ListenAndServe(":"+port, nil)
	if err != nil {
		log.Fatal("http.ListenAndServe : ", err)
	}
}
