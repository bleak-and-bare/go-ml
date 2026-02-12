import { useEffect, useMemo, useState, type ReactElement } from "react"
import ExecStatus from "./ExecStatus"
import { Group, ThemeIcon, Transition, Text, Paper, Table } from "@mantine/core"
import { IconBrain } from "@tabler/icons-react"
import { useWebSocket } from "./WebSocketContext"
import { Message } from "./Message"
import { formatDuration } from "./util"

type Stats = Extract<Message, { type: "stats" }>["data"]
type ExecStats = Extract<Message, { type: "exec_finished" }>["data"]

export function SystemInfo({ execStatus, curFolder }: { execStatus: ExecStatus, curFolder: string }): ReactElement {
    const ws = useWebSocket()
    const [execStats, setExecStats] = useState<ExecStats | null>(null)
    const [prevStats, setPrevStats] = useState<Stats | null>(null)
    const [stats, setStats] = useState<Stats | null>(null)

    const cpuUsage = useMemo(() => {
        if (stats === null || prevStats === null) return null

        if (prevStats.global === stats.global && prevStats.global === stats.global)
            return 0

        const procDiff = stats.process.user + stats.process.system - prevStats.process.user - prevStats.process.system
        const globDiff = stats.global + stats.global - prevStats.global - prevStats.global

        return Math.max(procDiff / globDiff * 100, 0)
    }, [prevStats, stats])

    const msgHandler = (data: string) => {
        try {
            const msg: Message = JSON.parse(data)
            switch (msg.type) {
                case "stats":
                    setStats(stats => {
                        setPrevStats(stats)
                        return msg.data
                    })
                    break
                case "exec_finished":
                    console.log(msg.data)
                    setExecStats(msg.data)
                    break
            }
        } catch (e) {
            console.error(`SystemInfo.msgHandler : ${e}`)
        }
    }

    useEffect(() => {
        if (ws.isConnected) {
            return ws.addSubscriber(msgHandler)
        }
    }, [ws.isConnected])

    return <>
        <Transition
            mounted={execStatus !== ExecStatus.STOPPED}
            transition="slide-left"
            duration={300}
            timingFunction="ease"
        >
            {styles => <Group style={styles} c={execStatus === ExecStatus.PAUSED ? "dimmed" : ""} >
                <ThemeIcon size="xs" variant="transparent"><IconBrain /></ThemeIcon>
                <Text>{curFolder} {execStatus === ExecStatus.RUNNING ? "running" : "paused"}</Text>
            </Group>
            }
        </Transition>
        {execStatus !== ExecStatus.STOPPED ? (
            <Paper withBorder p="xs" styles={{ root: { fontFamily: "monospace" } }}>
                {cpuUsage !== null ? <Text fz="sm">CPU Usage : {Math.round(cpuUsage)}%</Text> : <></>}
                {stats ? <>
                    <Text fz="sm">User usage : {Math.round(stats.process.user)}ms</Text>
                    <Text fz="sm">System usage : {Math.round(stats.process.system)}ms</Text>
                    <Text fz="sm">RSS : {Math.trunc(stats.rss * 100) / 100} kB</Text>
                    <Text fz="sm">Peak RSS : {Math.trunc(stats.peak_rss * 100) / 100} kB</Text>
                </> : <></>}
            </Paper>) : execStats !== null ? <>
                <Table variant="vertical" layout="fixed" captionSide="bottom" withTableBorder>
                    <Table.Tbody>
                        <Table.Tr>
                            <Table.Th>Elapsed</Table.Th>
                            <Table.Td>{formatDuration(execStats.duration)}</Table.Td>
                        </Table.Tr>
                        <Table.Tr>
                            <Table.Th>CPU Usage</Table.Th>
                            <Table.Td>{Math.floor((execStats.system_time + execStats.user_time) / execStats.duration * 100)}%</Table.Td>
                        </Table.Tr>
                        <Table.Tr>
                            <Table.Th>Exit status</Table.Th>
                            <Table.Td>{execStats.exit_status}</Table.Td>
                        </Table.Tr>
                    </Table.Tbody>
                    <Table.Caption>Execution stats</Table.Caption>
                </Table>
            </> : <></>}
    </>
}
