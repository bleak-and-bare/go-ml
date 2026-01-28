import "@mantine/core/styles.css";
import '@mantine/notifications/styles.css'
import { Text, AppShell, Burger, Title, Image, Group, MantineProvider, Box } from "@mantine/core";
import { theme } from "./theme";
import { WebSocketProvider } from "./WebSocketContext";
import { useDisclosure } from "@mantine/hooks";
import { Sidebar } from "./Sidebar";
import { Notifications } from "@mantine/notifications";

export default function App() {
    const [opened, { toggle }] = useDisclosure()

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
                    <Group h="100%" px="md" gap="sm">
                        <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
                        <Box>
                            <Image src="/icon/go.svg" alt="GO" width={52} height={52} fit="contain" />
                        </Box>
                        <Title order={2}>ML report</Title>
                    </Group>
                </AppShell.Header>
                <AppShell.Navbar p="md">
                    <Sidebar />
                </AppShell.Navbar>
                <AppShell.Main>
                    <Text>This is the main section, your app content here.</Text>
                    <Text>Layout used in most cases – Navbar and Header with fixed position</Text>
                </AppShell.Main>
            </AppShell>
        </WebSocketProvider>
    </MantineProvider>;
}
