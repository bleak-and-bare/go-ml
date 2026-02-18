import { Tabs, Text } from "@mantine/core";
import { LogType } from "../@types"
import Execution from "../component/Execution";
import { IconCode, IconCpu, IconTerminal } from "@tabler/icons-react";

type MainProps = {
    logFilter: Record<LogType, boolean>,
    clearLogs: boolean
}

enum Tab {
    Execution = "execution",
    Code = "code",
    REPL = "repl"
}

export default function MainSection({ logFilter, clearLogs }: MainProps) {
    return <Tabs variant="outline" radius="xs" defaultValue={Tab.Execution}>
        <Tabs.List>
            <Tabs.Tab value={Tab.Execution} leftSection={<IconCpu size={16} />}>
                Execution
            </Tabs.Tab>
            <Tabs.Tab value={Tab.Code} leftSection={<IconCode size={16} />}>
                Code
            </Tabs.Tab>
            <Tabs.Tab value={Tab.REPL} leftSection={<IconTerminal size={16} />}>
                REPL
            </Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel pt="sm" pl="xs" value={Tab.Execution}>
            <Execution logFilter={logFilter} clearLogs={clearLogs} />
        </Tabs.Panel>

        <Tabs.Panel pt="sm" pl="xs" value={Tab.Code}>
            <Text>Here goes the code</Text>
        </Tabs.Panel>

        <Tabs.Panel pt="sm" pl="xs" value={Tab.REPL}>
            <Text>Here goes the Go interpreter</Text>
        </Tabs.Panel>
    </Tabs>
}

