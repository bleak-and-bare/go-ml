package command

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
)

func CreateCmd(folder string) (*exec.Cmd, error) {
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

	cmd := exec.Command("go", "run", project)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	return cmd, nil
}
