import { Message } from "./Message"

type LogType = Extract<Message["type"], "info" | "error" | "table">
const AllLogType: LogType[] = ["info", "error", "table"]

export { AllLogType }
export type { LogType }
