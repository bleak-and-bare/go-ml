import { useEffect, useMemo, useRef, useState } from "react"
import { Box, Button, Divider, HoverCard, Stack, Text } from "@mantine/core"
import { Message, AllLogType, LogType } from "../@types"
import { useWebSocket } from "../context/WebSocketContext"
import { MessageItem } from "../component";
import { IconCode, IconPlus, IconSquareX, IconTrash } from "@tabler/icons-react";

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

    return <Stack gap="xs">
        {messagesToShow.length === 0
            ? <Text style={{ textAlign: "center" }}>Consider checking log filter or Run a program.</Text>
            : messagesToShow.map((msg, i) => msg.type === "exec_finished" ? <Divider key={i} variant="dotted" my="sm" />
                : <HoverCard key={i}>
                    <HoverCard.Target>
                        <Box>
                            <MessageItem message={msg} />
                        </Box>
                    </HoverCard.Target>

                    <HoverCard.Dropdown p="0">
                        <Button.Group>
                            <Button
                                size="xs"
                                variant="light"
                                id="test"
                                leftSection={<IconPlus size="14" />}>Text</Button>
                            <Button
                                size="xs"
                                variant="light"
                                disabled
                                leftSection={<IconCode size="14" />}>Code</Button>
                            <Button
                                onClick={() => setMessages(messages => messages.filter((_, k) => i !== k))}
                                size="xs"
                                variant="light"
                                leftSection={<IconTrash size="14" />}>Delete</Button>
                            <Button
                                onClick={() => setMessages(messages => messages.filter((_, k) => k > i))}
                                size="xs"
                                variant="light"
                                leftSection={<IconSquareX size="14" />}>Clear</Button>
                        </Button.Group>
                    </HoverCard.Dropdown>
                </HoverCard>)}
        <div ref={bottomRef} id="bottom-sentinel" />
    </Stack>
}

