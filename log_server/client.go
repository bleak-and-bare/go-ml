package main

import (
	"bytes"
	"fmt"
	"os"
	"time"

	"github.com/gorilla/websocket"
)

var (
	new_line = []byte{'\n'}
	space    = []byte{' '}
)

type Client struct {
	hub  *Hub
	conn *websocket.Conn
	send chan []byte
}

// create then register a new client
func NewClient(hub *Hub, conn *websocket.Conn) *Client {
	client := &Client{hub, conn, make(chan []byte, 256)}
	hub.Register(client)
	return client
}

func (c *Client) Close() {
	close(c.send)
}

func (c *Client) Send() chan<- []byte {
	return c.send
}

func (c *Client) unregister() {
	c.hub.Unregister(c)
	c.conn.Close()
}

// pumps message and push to the hub
func (c *Client) ReadPump() {
	defer c.unregister()

	c.conn.SetReadLimit(MAX_MSG_SIZE)
	c.conn.SetReadDeadline(time.Now().Add(PONG_WAIT))
	c.conn.SetPongHandler(func(string) error { // extend the deadline
		c.conn.SetReadDeadline(time.Now().Add(PONG_WAIT))
		return nil
	})

	for {
		_, msg, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseAbnormalClosure) {
				fmt.Fprintf(os.Stderr, "Client.ReadPump : %v\n", err)
			}
			return
		}

		msg = bytes.TrimSpace(bytes.ReplaceAll(msg, new_line, space))
		fmt.Println(string(msg))
		c.hub.Broadcast(msg)
	}
}

// pumps message from the hub to connection
func (c *Client) WritePump() {
	ticker := time.NewTicker(PING_PERIOD)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case msg, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(WRITE_WAIT))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := c.conn.NextWriter(websocket.TextMessage)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Client.WritePump : failed to get writer %v\n", err)
				return
			}
			w.Write(msg)

		msg_drain:
			for {
				select {
				case msg, ok := <-c.send:
					if !ok {
						break msg_drain
					}

					w.Write(new_line)
					w.Write(msg)

				default:
					break msg_drain
				}
			}

			if err := w.Close(); err != nil {
				fmt.Fprintf(os.Stderr, "Client.WritePump : failed to close writer %v\n", err)
				return
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(WRITE_WAIT))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				fmt.Fprintf(os.Stderr, "Client.WritePump : failed to write ping message %v\n", err)
				return
			}
		}
	}
}
