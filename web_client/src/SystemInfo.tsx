import { type ReactElement } from "react"
import ExecStatus from "./ExecStatus"
import { Group, ThemeIcon, Transition, Text } from "@mantine/core"
import { IconBrain } from "@tabler/icons-react"

export function SystemInfo({ execStatus, curFolder }: { execStatus: ExecStatus, curFolder: string }): ReactElement {
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
            </Group>}
        </Transition>
    </>
}
