package ws

import "github.com/bleak-and-bare/go-ml/message"

type Command struct {
	message.Message
	Client *Client
}
