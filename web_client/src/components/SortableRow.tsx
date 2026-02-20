import { Box, Menu, Group, ActionIcon } from "@mantine/core"
import { IconPlus, IconGripVertical, IconTextSize, IconCode, IconTrash, IconSquareX } from "@tabler/icons-react"
import { Message } from "../@types"
import { useRef, useState } from "react"
import MessageItem from "./MessageItem"
import { useSortable } from "@dnd-kit/react/sortable"
import classes from "../styles/SortableRow.module.css"

type SortableRowProps = {
    message: Message,
    index: number,
    onAddText: () => void,
    onAddCode?: () => void,
    onClear: () => void,
    onDelete: () => void,
}

function SortableRow({ message, index, onClear, onDelete, onAddText, onAddCode }: SortableRowProps) {
    const handleRef = useRef<HTMLButtonElement>(null)
    const { ref } = useSortable({ index, handle: handleRef, id: index })
    const [menuOpened, setMenuOpened] = useState(false)

    return <Group className={classes["sortable-row"]} ref={ref}>
        <Menu
            position="right-end"
            opened={menuOpened}
            onChange={setMenuOpened}
            withArrow
        >
            <Group className={classes["sortable-row-buttons"]} data-menu-opened={menuOpened}>
                <Menu.Target>
                    <ActionIcon
                        size="sm"
                        variant="subtle"
                    >
                        <IconPlus size={16} />
                    </ActionIcon>
                </Menu.Target>
                <ActionIcon
                    ref={handleRef}
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
        <Box flex={1} miw={0} style={{ cursor: "default" }}>
            <MessageItem message={message} />
        </Box>
    </Group>
}

export default SortableRow
