import "@mantine/core/styles.css";
import '@mantine/notifications/styles.css'
import '@mantine/code-highlight/styles.css'
import { AppShell, Burger, Title, Group, MantineProvider, ThemeIcon, Box } from "@mantine/core";
import { theme } from "./theme";
import { WebSocketProvider, GoCodeHighlightProvider } from "./contexts";
import { useDisclosure } from "@mantine/hooks";
import { Sidebar, MainSection } from "./sections";
import { Notifications } from "@mantine/notifications";
import { useState } from "react";
import { LogType } from "./@types";
import { IconChartCohort } from "@tabler/icons-react";
import classes from "./styles/App.module.css"

export default function App() {
    const [opened, { toggle }] = useDisclosure()
    const [clearLogs, setClearLogs] = useState(false)
    const [logFilter, setLogFilter] = useState<Record<LogType, boolean>>({
        error: true,
        info: true,
        table: true,
        progress: true,
    })

    return <MantineProvider
        defaultColorScheme="dark"
        theme={theme}
    >
        <Box className={classes.app}>
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
                            <ThemeIcon variant="transparent" color="cyan"><IconChartCohort /></ThemeIcon>
                            <Title order={3} c="cyan">Report</Title>
                        </Group>
                    </AppShell.Header>
                    <AppShell.Navbar p="md">
                        <Sidebar clearLogs={() => setClearLogs(c => !c)} setLogFilter={filter => setLogFilter(filter)} />
                    </AppShell.Navbar>
                    <AppShell.Main>
                        <GoCodeHighlightProvider>
                            <MainSection clearLogs={clearLogs} logFilter={logFilter} />
                        </GoCodeHighlightProvider>
                    </AppShell.Main>
                </AppShell>
            </WebSocketProvider>
        </Box>
    </MantineProvider>
}
