import { useEffect, useMemo, useRef, useState } from "react"
import { alpha, Box, Divider, Table, Text } from "@mantine/core"
import { Message } from "./Message"
import { useWebSocket } from "./WebSocketContext"
import { AllLogType, LogType } from "./LogType";

type MainProps = {
    logFilter: Record<LogType, boolean>,
    clearLogs: boolean
}

function Main({ logFilter, clearLogs }: MainProps) {
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
            if (msg.type === "exec_finished"
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

function MessageItem({ message }: { message: Message }) {
    switch (message.type) {
        case "info":
            return <Text>{message.data}</Text>
        case "error":
            return message.data.error.length === 0 ? <></> : <>
                <Divider color="red" my="xs" variant="dotted" />
                {message.data.stack_frame?.split('\n').filter(line => line.length > 0)
                    .map((line, i) => <Text key={i} fz="xs" c="red" styles={{ root: { fontFamily: "monospace" } }}>{line}</Text>)}
                {message.data.stack_frame ? <Text fz="xs" c="red" styles={{ root: { fontFamily: "monospace" } }}>...</Text> : <></>}
                <Box style={(theme) => ({
                    backgroundColor: alpha(theme.colors.red[3], 0.3),
                    color: theme.colors.red[3],
                    padding: '2px 6px',
                    borderRadius: 4,
                    display: 'inline-block',
                })}>
                    {message.data.error.split('\n').filter(line => line.length > 0).map((line, i) =>
                        <Text fz="sm" key={i}>{line}</Text>)}
                </Box>
            </>
        case "table":
            return <Table.ScrollContainer minWidth={360}>
                <Table data={message.data} />
            </Table.ScrollContainer>
        case "exec_finished":
            return <Divider variant="dotted" my="sm" />
    }

    return <></>
}


export default Main
