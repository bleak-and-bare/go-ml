FROM golang:1.25.5-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./

# get deps without changing requirements
RUN go mod download

COPY . .

# build without debug info
RUN go build -trimpath -ldflags="-s -w" -o logserver

# -------------------- runtime
FROM golang:1.25.5-alpine 

WORKDIR /app

COPY --from=builder /app/logserver /app/logserver

ENTRYPOINT [ "/app/logserver" ]
