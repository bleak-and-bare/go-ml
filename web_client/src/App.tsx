import "@mantine/core/styles.css";
import '@mantine/notifications/styles.css'
import { AppShell, Burger, Title, Group, MantineProvider, ThemeIcon } from "@mantine/core";
import { theme } from "./theme";
import { WebSocketProvider } from "./WebSocketContext";
import { useDisclosure } from "@mantine/hooks";
import { Sidebar } from "./Sidebar";
import { Notifications } from "@mantine/notifications";
import Main from "./MainSection";
import { useState } from "react";
import { LogType } from "./LogType";
import { IconRobot } from "@tabler/icons-react";

export default function App() {
    const [opened, { toggle }] = useDisclosure()
    const [clearLogs, setClearLogs] = useState(false)
    const [logFilter, setLogFilter] = useState<Record<LogType, boolean>>({
        error: true,
        info: true,
        table: true
    })

    return <MantineProvider
        defaultColorScheme="dark"
        theme={theme}
    >
        <WebSocketProvider>
            <AppShell
                withBorder={false}
                header={{ height: 60 }}
                navbar={{ width: 360, breakpoint: 'sm', collapsed: { mobile: !opened } }}
                padding="md"
            >
                <Notifications />
                <AppShell.Header>
                    <Group h="100%" px="md" gap="xs">
                        <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
                        <ThemeIcon variant="transparent" color="cyan"><IconRobot /></ThemeIcon>
                        <Title order={3} c="cyan">Report</Title>
                    </Group>
                </AppShell.Header>
                <AppShell.Navbar p="md">
                    <Sidebar clearLogs={() => setClearLogs(c => !c)} setLogFilter={filter => setLogFilter(filter)} />
                </AppShell.Navbar>
                <AppShell.Main>
                    <Main clearLogs={clearLogs} logFilter={logFilter} />
                </AppShell.Main>
            </AppShell>
        </WebSocketProvider>
    </MantineProvider>;
}
