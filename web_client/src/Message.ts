type CPUStat = {
    user: number, // milliseconds
    system: number, // milliseconds
}

type Message = {
    type: "execute" | "info" | "error",
    data: string,
} | {
    type: "abort" | "pause" | "resume"
} | {
    type: "exec_finished",
    data: {
        duration: number, // milliseconds
        user_time: number, // milliseconds
        system_time: number, // milliseconds
        exit_status: number // milliseconds
    }
} | {
    type: "stats",
    data: {
        process: CPUStat,
        global: number, // milliseconds
        rss: number, // kB
        peak_rss: number, // kB
        delta: number // milliseconds
    }
} | {
    type: "fulfilled",
    data: Exclude<Message["type"], "fulfilled">
}

type MessageType = Message["type"]

export type { MessageType, Message }
