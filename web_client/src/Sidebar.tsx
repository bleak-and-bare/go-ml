import { ActionIcon, Button, Divider, Group, Loader, MultiSelect, Select, Stack, Text, ThemeIcon, Title } from "@mantine/core"
import { IconLink, IconPlayerPause, IconPlayerPlay, IconRotateClockwise, IconTrash, IconUnlink, IconX } from "@tabler/icons-react"
import { useState, useEffect, type ReactElement, Dispatch, SetStateAction } from "react"
import { useWebSocket } from "./WebSocketContext"
import ExecStatus from "./ExecStatus"
import { Message } from "./Message"
import { notifications } from "@mantine/notifications"
import { SystemInfo } from "./SystemInfo"
import { AllLogType, LogType } from "./LogType"

type SidebarProps = {
    clearLogs: () => void,
    setLogFilter: Dispatch<SetStateAction<Record<LogType, boolean>>>
}

export function Sidebar({ clearLogs, setLogFilter }: SidebarProps): ReactElement {
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
        } finally {
            setLoading(false)
        }
    }

    const abortExecution = () => {
        ws.send(JSON.stringify({ type: "abort" }))
    }

    const resumeOrRunExec = () => {
        if (execStatus === ExecStatus.PAUSED) {
            ws.send(JSON.stringify({ type: "resume" }))
        } else if (curFolder.length > 0) {
            ws.send(JSON.stringify({
                type: "execute",
                data: curFolder
            }))
        } else {
            notifications.show({ color: 'red', title: 'No program to run', message: 'Select a program' })
        }
    }

    const pauseExecution = () => {
        ws.send(JSON.stringify({ type: "pause" }))
    }

    const msgHandler = (msgStr: string) => {
        try {
            const msg: Message = JSON.parse(msgStr)
            switch (msg.type) {
                case "exec_finished":
                    setExecStatus(ExecStatus.STOPPED)
                    break
                case "fulfilled":
                    switch (msg.data) {
                        case "execute":
                        case "resume":
                            setExecStatus(ExecStatus.RUNNING)
                            break
                        case "abort":
                            setExecStatus(ExecStatus.STOPPED)
                            break
                        case "pause":
                            setExecStatus(ExecStatus.PAUSED)
                            break
                    }
                    break
            }
        } catch (e) {
            console.error(`Sidebar.msgHandler : unknown message format : ${e}`)
        }
    }

    useEffect(() => {
        if (!ws.isConnected) {
            setExecStatus(ExecStatus.STOPPED)
        } else {
            fetchPlayground()
                .then(playground => setPlayground(playground))
                .catch(e => console.error(`ERROR fetchPlayground : ${e}`))
            return ws.addSubscriber(msgHandler)
        }
    }, [ws.isConnected])

    return <Stack gap="md">
        <Group gap="xs" justify="flex-end">
            <ThemeIcon variant="transparent" size="sm">
                {ws.isConnected ? <IconLink /> : <IconUnlink />}
            </ThemeIcon>
            <Text c="dimmed">{ws.isConnected ? "Connected" : "Disconnected"}</Text>
        </Group>
        <Group justify="space-between">
            <Title order={4}>Execution control</Title>
            <Group >
                <ActionIcon disabled={!ws.isConnected} onClick={abortExecution} variant="transparent">
                    <IconX />
                </ActionIcon>
                {execStatus === ExecStatus.RUNNING ? <ActionIcon onClick={pauseExecution} variant="transparent">
                    <IconPlayerPause />
                </ActionIcon> : <ActionIcon disabled={!ws.isConnected} onClick={resumeOrRunExec} variant="transparent">
                    <IconPlayerPlay />
                </ActionIcon>}
            </Group>
        </Group>
        <Group align="end">
            <Select
                flex={1}
                searchable
                label="Select program to run"
                placeholder="Type folder name"
                selectFirstOptionOnChange
                allowDeselect={false}
                data={playground}
                onSearchChange={setCurFolder}
                disabled={!ws.isConnected || execStatus !== ExecStatus.STOPPED}
            />
            {loading ? <Loader color="blue" type="dots" size="sm" /> :
                <ActionIcon onClick={() => fetchPlayground().then(playground => setPlayground(playground))} variant="subtle" title="Refetch">
                    <IconRotateClockwise />
                </ActionIcon>
            }
        </Group>
        <Divider size="xs" variant="dotted" />
        <SystemInfo execStatus={execStatus} curFolder={curFolder} />
        <MultiSelect
            label="Select log filters"
            placeholder="Select what to show"
            data={AllLogType}
            defaultValue={AllLogType}
            onChange={logs => setLogFilter(filter => {
                for (const log in filter) filter[log as LogType] = logs.includes(log)
                return { ...filter }
            })}
            clearable
        />
        <Button leftSection={<IconTrash size={16} />} variant="light" onClick={clearLogs}>Clear all logs</Button>
    </Stack>
}
