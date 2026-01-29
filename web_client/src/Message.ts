type Message = {
    type: "execute" | "info" | "error",
    data: string,
} | {
    type: "abort" | "pause" | "resume"
} | {
    type: "exec_finished",
    data: {
        duration: number,
        user_time: number,
        system_time: number,
        exit_status: number
    }
} | {
    type: "stats",
    data: {
        user_cpu: number,
        system_cpu: number,
        rss: number,
        peak_rss: number,
        delta: number
    }
}

type MessageType = Message["type"]

export type { MessageType, Message }
