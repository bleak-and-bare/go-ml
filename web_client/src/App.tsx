import { MantineProvider } from '@mantine/core'
import { WebSocketProvider } from './WebSocketContext'

function App() {
    return <MantineProvider>
        <WebSocketProvider>
        </WebSocketProvider>
    </MantineProvider>
}

export default App
