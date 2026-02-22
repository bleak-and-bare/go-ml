import { Divider, Box, Text, Table, Progress, Group } from "@mantine/core"
import { Message } from "../@types"
import Markdown from "react-markdown"
import { markdownComponents } from "./markdownComponents"
import classes from "../styles/MessageItem.module.css"
import { useEffect, useState } from "react"

export default function MessageItem({ message }: { message: Message }) {
    switch (message.type) {
        case "info":
            return <Markdown components={markdownComponents}>{message.data}</Markdown>

        case "error":
            return message.data.error.length === 0 ? <></> : <>
                <Divider color="red" my="xs" variant="dotted" />
                {message.data.stack_frame?.split('\n').filter(line => line.length > 0)
                    .map((line, i) => <Text key={i} fz="xs" c="red" styles={{ root: { fontFamily: "monospace" } }}>{line}</Text>)}
                {message.data.stack_frame ? <Text fz="xs" c="red" styles={{ root: { fontFamily: "monospace" } }}>...</Text> : <></>}
                <Box className={classes["error-container"]}>
                    {message.data.error.split('\n').filter(line => line.length > 0).map((line, i) =>
                        <Text fz="sm" key={i}>{line}</Text>)}
                </Box>
            </>

        case "table":
            return <Table.ScrollContainer minWidth={480} maxHeight={512}>
                <Table data={message.data} />
            </Table.ScrollContainer>

        case "progress":
            return <ProgressRow
                running={message.running === true}
                label={message.data.label}
                value={message.data.value}
            />
    }

    return <></>
}

function ProgressRow({ running, value, label }: {
    value: number,
    label: string,
    running: boolean
}) {
    const spinnerFrames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    const frameDuration = 80
    const [curFrame, setCurFrame] = useState(0)

    useEffect(() => {
        const interval = setInterval(() => setCurFrame(i => (i + 1) % spinnerFrames.length), frameDuration)
        return () => clearInterval(interval)
    }, [])

    return <Group>
        {running && <Text>{spinnerFrames[curFrame]}</Text>}
        <Markdown components={markdownComponents}>
            {label}
        </Markdown>
        <Progress
            flex={1}
            color="indigo"
            size="sm"
            value={value}
            style={{ maxWidth: "480px" }}
            transitionDuration={200}
            animated={value < 100}
        />
    </Group>
}
