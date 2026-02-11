function formatDuration(ms: number) {
    let s = Math.floor(ms / 1000)
    ms %= 1000

    let m = Math.floor(s / 60)
    s %= 60

    let h = Math.floor(m / 60)
    m %= 60

    return `${h}:${m.toString().padStart(2, "0")}:${s
        .toString()
        .padStart(2, "0")}.${ms.toString().padStart(3, "0")}`;
}

export { formatDuration }
