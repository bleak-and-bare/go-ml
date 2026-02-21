import { CodeHighlightAdapterProvider, createHighlightJsAdapter } from "@mantine/code-highlight";
import golang from "highlight.js/lib/languages/go"
import hljs from "highlight.js";

export function GoCodeHighlightProvider({ children }: { children: React.ReactNode }) {
    hljs.registerLanguage('go', golang)
    const highlightGoAdapter = createHighlightJsAdapter(hljs)
    return <CodeHighlightAdapterProvider adapter={highlightGoAdapter}>{children}</CodeHighlightAdapterProvider>
}
