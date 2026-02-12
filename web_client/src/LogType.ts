import { Message } from "./Message"

type LogType = Extract<Message["type"], "info" | "error">
const AllLogType: LogType[] = ["info", "error"]

export { AllLogType }
export type { LogType }
