import { Message } from "./Message"

type LogType = Extract<Message["type"], "info" | "error" | "progress" | "table">
const AllLogType: LogType[] = ["info", "error", "table", "progress"]

export { AllLogType }
export type { LogType }
