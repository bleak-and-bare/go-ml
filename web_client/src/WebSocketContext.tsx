/* eslint-disable react-refresh/only-export-components */
import type React from "react";
import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";

type MessageHandler = (msg: string) => void

interface WSContext {
    send: (data: string) => void
    addSubscriber: (h: MessageHandler) => () => void
    isConnected: () => boolean
}

const WebSocketContext = createContext<WSContext | null>(null)

export function useWebSocket() {
    const ctx = useContext(WebSocketContext)
    if (!ctx) {
        throw new Error("useWebSocket hook must be used within provider")
    }
    return ctx
}

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
    const socketRef = useRef<WebSocket | null>(null)
    const handlersRef = useRef<Set<MessageHandler>>(new Set())
    const [connected, setConnected] = useState(false)

    useEffect(() => {
        const wsAddress: string = import.meta.env.VITE_SERVER || "localhost:8000"
        const ws = new WebSocket(`ws://${wsAddress}/ws`)
        socketRef.current = ws

        ws.onopen = () => setConnected(true)
        ws.onclose = () => setConnected(false)
        ws.onmessage = (event) => {
            for (const handler of handlersRef.current) {
                handler(event.data)
            }
        }

        return () => ws.close()
    }, [])

    const send = useCallback((message: string) => {
        if (socketRef.current?.readyState === WebSocket.OPEN)
            socketRef.current.send(message)
    }, [])

    const addSubscriber = useCallback((handler: MessageHandler) => {
        handlersRef.current.add(handler)
        return () => handlersRef.current.delete(handler)
    }, [])

    return <WebSocketContext.Provider value={{
        send,
        addSubscriber,
        isConnected: () => connected,
    }}>
        {children}
    </WebSocketContext.Provider>
}
