package message

type ErrorStruct struct {
	Error      string `json:"error"`
	StackFrame string `json:"stack_frame,omitempty"`
}
