package message

type TableStruct struct {
	Caption string     `json:"caption,omitempty"`
	Head    []string   `json:"head"`
	Body    [][]string `json:"body"`
}
