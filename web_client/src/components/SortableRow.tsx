import { Box, Menu, Group, ActionIcon, Button, Divider } from "@mantine/core"
import { IconPlus, IconGripVertical, IconTextSize, IconCode, IconTrash, IconSquareX } from "@tabler/icons-react"
import { Message } from "../@types"
import { useEffect, useRef, useState } from "react"
import MessageItem from "./MessageItem"
import { useSortable } from "@dnd-kit/react/sortable"
import classes from "../styles/SortableRow.module.css"
import { theme } from "../theme"

type SortableRowProps = {
    message: Message,
    id: number,
    index: number,
    onAddText: () => void,
    onAddCode?: () => void,
    onClear: () => void,
    onDelete: () => void,
}

function SortableRow({ message, id, index, onClear, onDelete, onAddText, onAddCode }: SortableRowProps) {
    const defaultHandleRef = useRef<HTMLButtonElement>(null)
    const largeHandleRef = useRef<HTMLButtonElement>(null)
    const { ref, handleRef } = useSortable({ index, id })
    const [menuOpened, setMenuOpened] = useState(false)

    useEffect(() => {
        const mq = window.matchMedia(`(min-width: ${theme.breakpoints!.sm})`)
        const updateRef = () => {
            const ref = mq.matches
                ? largeHandleRef.current
                : defaultHandleRef.current
            handleRef(ref)
        }

        updateRef()
        mq.addEventListener('change', updateRef)

        return () => mq.removeEventListener('change', updateRef)
    }, [])

    return message.type === "exec_finished"
        ? <Divider variant="dotted" my="sm" ref={ref} />
        : <Group className={classes["sortable-row"]} ref={ref}>
            <Menu
                position="right-end"
                opened={menuOpened}
                onChange={setMenuOpened}
                withArrow
            >
                <Group className={classes["sortable-row-side-buttons"]} data-menu-opened={menuOpened}>
                    <Menu.Target>
                        <ActionIcon
                            size="sm"
                            variant="subtle"
                        >
                            <IconPlus size={16} />
                        </ActionIcon>
                    </Menu.Target>
                    <ActionIcon
                        ref={largeHandleRef}
                        size="sm"
                        variant="subtle"
                        style={{ cursor: "grab" }}
                    >
                        <IconGripVertical size={16} />
                    </ActionIcon>
                </Group>

                <Menu.Dropdown>
                    <Menu.Item fz="xs" onClick={onAddText} leftSection={<IconTextSize size={14} />}>Text</Menu.Item>
                    <Menu.Item fz="xs" disabled onClick={onAddCode} leftSection={<IconCode size={14} />}>Code</Menu.Item>
                    <Menu.Item fz="xs" onClick={onDelete} leftSection={<IconTrash size={14} />}>Delete</Menu.Item>
                    {index > 0 && <Menu.Item fz="xs" onClick={onClear} leftSection={<IconSquareX size={14} />}>Clear</Menu.Item>}
                </Menu.Dropdown>
            </Menu>

            <Group gap="xs" className={classes["sortable-row-center-buttons"]}>
                <ActionIcon
                    variant="default"
                    ref={defaultHandleRef}
                    style={{ cursor: "grab" }}
                >
                    <IconGripVertical size={14} />
                </ActionIcon>
                <Button.Group>
                    <Button
                        onClick={onAddText}
                        variant="default"
                        size="sm"
                        px="xs"
                    >
                        <IconTextSize size={14} />
                    </Button>
                    <Button
                        onClick={onAddCode}
                        variant="default"
                        disabled
                        size="sm"
                        px="xs"
                    >
                        <IconCode size={14} />
                    </Button>
                    <Button
                        onClick={onDelete}
                        variant="default"
                        size="sm"
                        px="xs"
                    >
                        <IconTrash size={14} />
                    </Button>
                    {index > 0 && <Button
                        onClick={onClear}
                        variant="default"
                        size="sm"
                        px="xs">
                        <IconSquareX size={14} />
                    </Button>}
                </Button.Group>
            </Group>

            <Box flex={1} miw={0} style={{ cursor: "default" }}>
                <MessageItem message={message} />
            </Box>
        </Group>
}

export default SortableRow
