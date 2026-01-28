import { ActionIcon, Autocomplete, Divider, Group, Loader, Stack, Text, ThemeIcon, Title } from "@mantine/core"
import { IconLink, IconPlayerPause, IconPlayerPlay, IconRotateClockwise, IconUnlink, IconX } from "@tabler/icons-react"
import { useState, useEffect, type ReactElement } from "react"
import { useWebSocket } from "./WebSocketContext"
import ExecStatus from "./ExecStatus"
import MessageType from "./MessageType"
import { notifications } from "@mantine/notifications"
import { SystemInfo } from "./SystemInfo"

export function Sidebar(): ReactElement {
    const ws = useWebSocket()
    const [playground, setPlayground] = useState<string[]>([])
    const [loading, setLoading] = useState(true)
    const [curFolder, setCurFolder] = useState("")
    const [execStatus, setExecStatus] = useState(ExecStatus.STOPPED)

    const fetchPlayground = async () => {
        const serverAddr: string = import.meta.env.VITE_SERVER || "localhost:8000"
        try {
            setLoading(true)
            const res = await fetch(`http://${serverAddr}/api/playgrounds`)
            if (!res.ok) {
                throw new Error("Sidebar.fetchPlayground : http error : " + res.status)
            }
            const data: { folders: string[] } = await res.json()
            return data.folders
        } catch (e) {
            // TODO : display error
            console.error("weehoo : " + e)
        } finally {
            setLoading(false)
        }
        return []
    }

    const abortExecution = () => {
        setExecStatus(ExecStatus.STOPPED)
        ws.send(JSON.stringify({ type: "abort" }))
    }

    const resumeOrRunExec = () => {
        if (execStatus === ExecStatus.PAUSED) {
            setExecStatus(ExecStatus.RUNNING)
            // ws.send(JSON.stringify({ type: "resume" }))
        } else if (curFolder.length > 0) {
            setExecStatus(ExecStatus.RUNNING)
            // ws.send(JSON.stringify({
            //     type: "execute",
            //     data: curFolder
            // }))
        } else {
            notifications.show({ color: 'red', title: 'No program to run', message: 'Select a program' })
        }
    }

    const pauseExecution = () => {
        setExecStatus(ExecStatus.PAUSED)
        ws.send(JSON.stringify({ type: "resume" }))
    }

    const msgHandler = (msgStr: string) => {
        try {
            const msg: { type: string, data?: string } = JSON.parse(msgStr)
            switch (msg.type) {
                case MessageType.EXEC_FINISHED:
                    setExecStatus(ExecStatus.STOPPED)
                    break
            }
        } catch (e) {
            console.error(`Sidebar.msgHandler : unknown message format : ${e}`)
        }
    }

    useEffect(() => {
        fetchPlayground().then(playground => setPlayground(playground))
        const unsubscribe = ws.addSubscriber(msgHandler)
        return unsubscribe
    }, [])

    return <Stack gap="md">
        <Group gap="xs" justify="flex-end">
            <ThemeIcon variant="transparent" size="sm">
                {ws.isConnected() ? <IconLink /> : <IconUnlink />}
            </ThemeIcon>
            <Text c="dimmed">{ws.isConnected() ? "Connected" : "Disconnected"}</Text>
        </Group>
        <Group justify="space-between">
            <Title order={4}>Execution control</Title>
            <Group>
                <ActionIcon onClick={abortExecution} variant="transparent">
                    <IconX />
                </ActionIcon>
                {execStatus === ExecStatus.RUNNING ? <ActionIcon onClick={pauseExecution} variant="transparent">
                    <IconPlayerPause />
                </ActionIcon> : <ActionIcon onClick={resumeOrRunExec} variant="transparent">
                    <IconPlayerPlay />
                </ActionIcon>}
            </Group>
        </Group>
        <Group align="end">
            <Autocomplete
                flex={1}
                label="Select program to run"
                placeholder="Type folder name"
                selectFirstOptionOnChange
                data={playground}
                onChange={setCurFolder}
                disabled={execStatus !== ExecStatus.STOPPED}
            />
            {loading ? <Loader color="blue" type="dots" size="sm" /> :
                <ActionIcon onClick={() => fetchPlayground().then(playground => setPlayground(playground))} variant="subtle" title="Refetch">
                    <IconRotateClockwise />
                </ActionIcon>
            }
        </Group>
        <Divider size="xs" variant="dotted" />
        <SystemInfo execStatus={execStatus} curFolder={curFolder} />
    </Stack>
}
