/* eslint-disable react-refresh/only-export-components */
import type React from "react";
import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";

type MessageHandler = (msg: string) => void

interface WSContext {
    send: (data: string) => void
    addSubscriber: (h: MessageHandler) => () => void
    close: () => void
    isConnected: boolean
}

const WebSocketContext = createContext<WSContext | null>(null)

function useWebSocket() {
    const ctx = useContext(WebSocketContext)
    if (!ctx) {
        throw new Error("useWebSocket hook must be used within provider")
    }
    return ctx
}

function WebSocketProvider({ children }: { children: React.ReactNode }) {
    const MAX_CONN_ATEMPT = 10
    const socketRef = useRef<WebSocket | null>(null)
    const [connected, setConnected] = useState(false)

    const wsConnect = async (): Promise<WebSocket> => {
        return new Promise((res, rej) => {
            const wsAddress: string = import.meta.env.VITE_SERVER || "localhost:8000"
            const ws = new WebSocket(`ws://${wsAddress}/ws`)

            ws.onopen = () => {
                setConnected(true)
                res(ws)
            }

            ws.onerror = (err) => {
                rej(err)
            }

            ws.onclose = () => setConnected(false)
        })
    }

    useEffect(() => {
        let reconnectTask = -1
        if (!connected) {
            let attempt = 0
            reconnectTask = setInterval(() => {
                if (attempt >= MAX_CONN_ATEMPT) {
                    clearInterval(reconnectTask)
                    return
                }

                attempt++
                wsConnect().then(ws => {
                    socketRef.current = ws
                    clearInterval(reconnectTask)
                }).catch(err => console.error(err))
            }, 3000)

            return () => clearInterval(reconnectTask)
        }

        return () => {
            if (reconnectTask > 0) clearInterval(reconnectTask)
            socketRef.current?.close()
        }
    }, [connected])

    const send = useCallback((message: string) => {
        if (socketRef.current?.readyState === WebSocket.OPEN)
            socketRef.current.send(message)
    }, [])

    const close = useCallback(() => {
        if (socketRef.current?.readyState === WebSocket.OPEN)
            socketRef.current.close()
    }, [])

    const addSubscriber = useCallback((handler: MessageHandler) => {
        const cb = (e: MessageEvent) => handler(e.data)
        socketRef.current?.addEventListener('message', cb)
        return () => socketRef.current?.removeEventListener('message', cb)
    }, [])

    return <WebSocketContext.Provider value={{
        send,
        close,
        addSubscriber,
        isConnected: connected,
    }}>
        {children}
    </WebSocketContext.Provider>
}

export { useWebSocket, WebSocketProvider }
