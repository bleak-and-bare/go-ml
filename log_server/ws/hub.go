package ws

import (
	"fmt"
	"sync"
	"time"
)

type Hub struct {
	mutex sync.Mutex

	// registered clients
	clients map[*Client]bool

	// message from clients
	broadcast chan []byte

	register   chan *Client
	unregister chan *Client
}

func NewHub() *Hub {
	return &Hub{
		clients:    make(map[*Client]bool),
		broadcast:  make(chan []byte),
		register:   make(chan *Client),
		unregister: make(chan *Client),
	}
}

func (h *Hub) Register(c *Client) { h.register <- c }

func (h *Hub) Unregister(c *Client) { h.unregister <- c }

func (h *Hub) Broadcast(msg []byte) { h.broadcast <- msg }

func (h *Hub) NotifyClients(n func(*Client, time.Duration), interval time.Duration) {
	ticker := time.NewTicker(interval)
	var prev time.Time

	for t := range ticker.C {
		h.mutex.Lock()
		var dt time.Duration
		if !prev.IsZero() {
			dt = time.Since(prev)
		}

		for client := range h.clients {
			n(client, dt)
		}
		h.mutex.Unlock()
		prev = t
	}
}

func (h *Hub) Run() {
	for {
		select {
		case client := <-h.register:
			h.mutex.Lock()
			fmt.Printf("Hub.Run : new client %p\n", client)
			h.clients[client] = true
			h.mutex.Unlock()
		case client := <-h.unregister:
			h.mutex.Lock()
			if _, ok := h.clients[client]; ok {
				fmt.Printf("Hub.Run : %p unregistered\n", client)
				delete(h.clients, client)
			}
			h.mutex.Unlock()
		case message := <-h.broadcast:
			for client := range h.clients {
				client.Send() <- message
			}
		}
	}
}
