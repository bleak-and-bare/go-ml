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

export function useWebSocket() {
    const ctx = useContext(WebSocketContext)
    if (!ctx) {
        throw new Error("useWebSocket hook must be used within provider")
    }
    return ctx
}

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
    const MAX_CONN_ATEMPT = 10
    const socketRef = useRef<WebSocket | null>(null)
    const handlersRef = useRef<Set<MessageHandler>>(new Set())
    const [connected, setConnected] = useState(false)

    const wsConnect = async (): Promise<WebSocket> => {
        return new Promise((res, rej) => {
            const wsAddress: string = import.meta.env.VITE_SERVER || "localhost:8000"
            const ws = new WebSocket(`ws://${wsAddress}/ws`)

            ws.onopen = () => {
                setConnected(true)
                return res(ws)
            }

            ws.onerror = (err) => {
                rej(err)
            }

            ws.onclose = () => setConnected(false)
            ws.onmessage = (event) => {
                for (const handler of handlersRef.current) {
                    handler(event.data)
                }
            }
        })
    }

    useEffect(() => {
        if (!connected) {
            let attempt = 0
            const i = setInterval(() => {
                if (attempt >= MAX_CONN_ATEMPT) {
                    clearInterval(i)
                    return
                }

                attempt++
                wsConnect().then(ws => {
                    socketRef.current = ws
                    clearInterval(i)
                }).catch(err => console.error(err))
            }, 3000)

            return () => clearInterval(i)
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
        handlersRef.current.add(handler)
        return () => handlersRef.current.delete(handler)
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
