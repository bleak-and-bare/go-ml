import { Divider, Box, Text, alpha, Table, Progress, Group } from "@mantine/core"
import { Message } from "../@types"
import Markdown from "react-markdown"

export default function MessageItem({ message }: { message: Message }) {
    switch (message.type) {
        case "info":
            return <Markdown>{message.data}</Markdown>

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

        case "progress":
            return <Group>
                <Text>{message.data.label}</Text>
                <Progress
                    flex={1}
                    color="indigo"
                    size="sm"
                    value={message.data.value}
                    style={{ maxWidth: "480px" }}
                    transitionDuration={200}
                    animated={message.data.value < 100}
                />
            </Group>

        case "exec_finished":
            return <Divider variant="dotted" my="sm" />
    }

    return <></>
}
