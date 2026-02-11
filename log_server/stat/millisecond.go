package stat

import "time"

type Millisecond int64

func (m *Millisecond) Duration() time.Duration {
	return time.Duration(*m) * time.Millisecond
}
