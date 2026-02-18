import { useEffect, useMemo, useRef, useState } from "react"
import { Text } from "@mantine/core"
import { Message, AllLogType, LogType } from "../@types"
import { useWebSocket } from "../context/WebSocketContext"
import { MessageItem } from "../component";

type MainProps = {
    logFilter: Record<LogType, boolean>,
    clearLogs: boolean
}

export default function MainSection({ logFilter, clearLogs }: MainProps) {
    const ws = useWebSocket()
    const bottomRef = useRef<HTMLDivElement | null>(null)
    const [messages, setMessages] = useState<Message[]>([])
    const messagesToShow = useMemo(() => messages
        .filter(msg => logFilter[msg.type as LogType] || msg.type === "exec_finished"),
        [messages, logFilter])

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [messages])

    useEffect(() => {
        setMessages([])
    }, [clearLogs])

    const msgHandler = (data: string) => {
        try {
            const msg: Message = JSON.parse(data)
            if (msg.type === "progress") {
                setMessages(messages => {
                    let update = false
                    messages = messages.map(prev => {
                        if (prev.type === "progress" && prev.data.id === msg.data.id) {
                            update = true
                            return msg
                        }
                        return prev
                    })

                    return update ? messages : [...messages, msg]
                })
            } else if (msg.type === "exec_finished"
                || AllLogType.includes(msg.type as LogType))
                setMessages(messages => [...messages, msg])
        } catch (e) {
            setMessages(messages => [...messages, { type: "error", data: { error: `${e}` } }])
            // console.error(`Main.msgHandler : ${e}`)
        }
    }

    useEffect(() => {
        if (ws.isConnected) {
            return ws.addSubscriber(msgHandler)
        }
    }, [ws.isConnected])

    return <>
        {messagesToShow.length === 0
            ? <Text styles={{ root: { textAlign: "center" } }}>Consider checking log filter or Run a program.</Text>
            : messagesToShow.map((msg, i) => <MessageItem message={msg} key={i} />)}
        <div ref={bottomRef} />
    </>
}

