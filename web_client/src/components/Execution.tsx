import { useEffect, useMemo, useRef, useState } from "react"
import { Divider, Stack, Text } from "@mantine/core"
import { Message, AllLogType, LogType } from "../@types"
import { useWebSocket } from "../contexts/WebSocketContext"
import SortableRow from "./SortableRow";
import { DragDropProvider } from "@dnd-kit/react";

export default function Execution({ logFilter, clearLogs }: {
    logFilter: Record<LogType, boolean>,
    clearLogs: boolean
}) {
    const ws = useWebSocket()
    const bottomRef = useRef<HTMLDivElement | null>(null)
    const [messages, setMessages] = useState<Message[]>([])
    const messagesToShow = useMemo(() => messages
        .filter(msg => logFilter[msg.type as LogType] || msg.type === "exec_finished"), // using type="exec_finished" to define an execution delimiter
        [messages, logFilter])

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" })

        if (messages.length === 1 && messages[0].type === "exec_finished") {
            setMessages([])
            return
        }

        const toRemove: number[] = []
        for (let i = 0; i < messages.length - 1; ++i) {
            if (messages[i].type !== "exec_finished") continue

            if (i === 0 || messages[i].type === messages[i + 1].type) {
                toRemove.push(i)
            }
        }

        if (toRemove.length > 0)
            setMessages(messages.filter((_, i) => toRemove.indexOf(i) < 0))
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

    return <DragDropProvider onDragEnd={(_event) => {
        // console.log({ index: event.operation.source.initialIndex })
        // setMessages(messages => (move(messages, event) as unknown as Message[]))
    }}>
        <Stack gap="xs">
            {messagesToShow.length === 0
                ? <Text style={{ textAlign: "center" }}>Consider checking log filter or Run a program.</Text>
                : messagesToShow.map((msg, i) => msg.type === "exec_finished"
                    ? <Divider key={i} variant="dotted" my="sm" />
                    : <SortableRow
                        index={i}
                        onClear={() => setMessages(messages => messages.filter((_, k) => k > i))}
                        onDelete={() => setMessages(messages => messages.filter((_, k) => k !== i))}
                        onAddText={() => { }}
                        key={i} message={msg}
                    />
                )}
            <div ref={bottomRef} id="bottom-sentinel" />
        </Stack>
    </DragDropProvider>
}
