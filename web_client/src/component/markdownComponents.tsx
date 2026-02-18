import { List, Text, Title } from "@mantine/core";
import { Components } from "react-markdown";

export const markdownComponents: Components = {
    h1: ({ children }) => <Title order={1} size="h2">{children}</Title>,
    h2: ({ children }) => <Title order={2} size="h3">{children}</Title>,
    h3: ({ children }) => <Title order={3} size="h4">{children}</Title>,

    p: Text,
    ul: ({ children }) => <List withPadding>{children}</List>,
    ol: ({ children }) => <List withPadding type="ordered">{children}</List>,
    li: ({ children }) => <List.Item>{children}</List.Item>
}
