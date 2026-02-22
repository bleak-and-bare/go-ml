import { useEffect, useMemo, useRef, useState } from "react"
import { ActionIcon, Divider, Stack, Text } from "@mantine/core"
import { Message, AllLogType, LogType } from "../@types"
import { useWebSocket } from "../contexts/WebSocketContext"
import { IconArrowNarrowUp, IconTrash } from "@tabler/icons-react";
import appClasses from "../styles/App.module.css"
import classes from "../styles/Execution.module.css"
import MessageItem from "./MessageItem";

export default function Execution({ logFilter, clearLogs }: {
    logFilter: Record<LogType, boolean>,
    clearLogs: boolean
}) {
    const ws = useWebSocket()
    const bottomRef = useRef<HTMLDivElement | null>(null)
    const [messages, setMessages] = useState<Message[]>([])
    const [displayScrollBtn, setDisplayScrollBtn] = useState(false)
    const messagesToShow = useMemo(() => messages
        .filter(msg => logFilter[msg.type as LogType] || msg.type === "exec_finished"), // using type="exec_finished" to define an execution delimiter
        [messages, logFilter])

    useEffect(() => {
        const shouldDisplay = shouldDisplayArrowUp()
        setDisplayScrollBtn(shouldDisplay)
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
                msg.running = true
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
                || AllLogType.includes(msg.type as LogType)) {
                setMessages(messages => {
                    if (msg.type === "exec_finished")
                        messages = messages.map(msg => {
                            if (msg.type === "progress")
                                return {
                                    ...msg,
                                    running: false
                                }

                            return msg
                        })

                    return [...messages, msg]
                })
            }
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

    const appClass = `.${appClasses.app}`

    const shouldDisplayArrowUp = () => {
        const appContainer = document.querySelector(appClass) as HTMLDivElement
        return appContainer !== null && appContainer.clientHeight < appContainer.scrollHeight
    }

    const scrollUp = () => {
        const appContainer = document.querySelector(appClass)
        appContainer?.scrollTo({ top: 0, behavior: 'smooth' })
    }

    return <>
        <Stack gap="xs">
            {messagesToShow.length === 0
                ? <Text style={{ textAlign: "center" }}>Consider checking log filter or Run a program.</Text>
                : messagesToShow.map((msg, i) => msg.type === "exec_finished"
                    ? <Divider key={i} variant="dotted" my="sm" />
                    : <MessageItem key={i} message={msg} />
                )}
            <div className={classes["scroll-up-btn"]} data-show={displayScrollBtn}>
                <ActionIcon
                    onClick={scrollUp}
                    variant="light"
                    radius="lg"
                    size="lg"
                >
                    <IconArrowNarrowUp size={24} />
                </ActionIcon>
            </div>
            <div ref={bottomRef} id="bottom-sentinel" />
        </Stack>
        {messages.length > 0 && <ActionIcon
            onClick={() => setMessages([])}
            variant="light"
            radius="lg"
            size="lg"
            className={classes['clear-btn']}
        >
            <IconTrash size={24} />
        </ActionIcon>}
    </>
}
