package command

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"

	"github.com/bleak-and-bare/go-ml/log_server/stat"
	"github.com/bleak-and-bare/go-ml/log_server/ws"
	"github.com/bleak-and-bare/go-ml/message"
)

func CreateGoRunCmd(folder string) (*exec.Cmd, error) {
	src_path := "/app/playground"
	if path, ok := os.LookupEnv("MAPPED_PLAYGROUND_PATH"); ok {
		src_path = path
	}

	project := filepath.Join(src_path, folder)
	info, err := os.Stat(project)
	if err != nil {
		return nil, err
	}

	if !info.IsDir() {
		return nil, fmt.Errorf("CreateCommand : %v is not a folder inside %v", folder, os.Getenv("PLAYGROUND_PATH"))
	}

	cmd := exec.Command("go", "run", ".")
	cmd.Dir = project
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	return cmd, nil
}

func send_info_to_client(client *ws.Client, msg string) {
	msg_bytes, _ := json.Marshal(message.Message{
		Type: message.INFO,
		Data: msg,
	})

	select {
	case <-client.Context().Done():
	case client.Send() <- msg_bytes:
	}
}

func send_error_to_client(client *ws.Client, err error) {
	err_bytes, _ := json.Marshal(message.Message{
		Type: message.ERROR,
		Data: err.Error(),
	})

	select {
	case <-client.Context().Done():
	case client.Send() <- err_bytes:
	}
}

func notify_req_fulfilled_to_client(req message.MessageType, client *ws.Client) {
	msg_bytes, _ := json.Marshal(message.Message{
		Type: message.FULFILLED,
		Data: req,
	})

	select {
	case <-client.Context().Done():
	case client.Send() <- msg_bytes:
	}
}

func notify_exec_finished_to_client(start time.Time, client *ws.Client) {
	exec_cmd := client.GetExecCmd()
	ps := exec_cmd.ProcessState

	msg_bytes, _ := json.Marshal(message.Message{
		Type: message.EXEC_FINISHED,
		Data: stat.ProcessStat{
			Duration:   stat.Millisecond(time.Since(start).Milliseconds()),
			UserTime:   stat.Millisecond(ps.UserTime().Milliseconds()),
			SystemTime: stat.Millisecond(ps.SystemTime().Milliseconds()),
			ExitStatus: ps.ExitCode(),
		},
	})

	select {
	case <-client.Context().Done():
	case client.Send() <- msg_bytes:
	}
}

func send_frame_to_client(bytes []byte, client *ws.Client, msg_type message.MessageType) {
	var msg message.Message
	if err := json.Unmarshal(bytes, &msg); err != nil {
		msg.Type = msg_type
		msg.Data = string(bytes)
	}

	b, _ := json.Marshal(msg)

	select {
	case <-client.Context().Done():
	case client.Send() <- b:
	}
}

func stream_output(cmd *exec.Cmd, client *ws.Client) error {
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}

	stdio, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}

	go func() {
		scanner := bufio.NewScanner(stdio)
		for scanner.Scan() {
			send_frame_to_client(scanner.Bytes(), client, message.INFO)
		}
	}()

	go func() {
		var buf bytes.Buffer
		_, err := io.Copy(&buf, stderr)
		if err != nil {
			send_error_to_client(client, err)
			return
		}

		send_frame_to_client(buf.Bytes(), client, message.ERROR)
	}()

	return nil
}
