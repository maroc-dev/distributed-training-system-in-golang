package main

import (
	"net"
	"net/rpc"
	"testing"
	"time"
)

func startDistWorker(t *testing.T) (string, func()) {
	t.Helper()
	addr, stop, err := RunWorker("127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(stop)
	return addr, stop
}

func TestWorkerRPC(t *testing.T) {
	// Two services must coexist without using the global RPC server.
	for workerID := 0; workerID < 2; workerID++ {
		addr, stop := startDistWorker(t)
		client, err := rpc.Dial("tcp", addr)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { client.Close() })
		var pong PingReply
		if err := client.Call("Worker.Ping", PingArgs{}, &pong); err != nil || !pong.OK {
			t.Fatalf("Ping: OK=%v, error=%v", pong.OK, err)
		}
		args := GradArgs{W: Init(1), Batch: NewBatcher(1, 32).Next(0)}
		var reply GradReply
		if err := client.Call("Worker.Grad", args, &reply); err != nil {
			t.Fatal(err)
		}
		assertSeqWeightsEqual(t, reply.Grad, Grad(args.W, args.Batch))
		for _, invalid := range []GradArgs{
			{W: nil, Batch: args.Batch},
			{W: args.W, Batch: nil},
			{W: args.W, Batch: []Example{{X: []float64{1}}}},
		} {
			if err := client.Call("Worker.Grad", invalid, &reply); err == nil {
				t.Fatal("invalid gradient request succeeded")
			}
		}
		stop() // Shutdown also closes an otherwise idle client connection.
		stop()
		conn, err := net.DialTimeout("tcp", addr, 100*time.Millisecond)
		if err == nil {
			conn.Close()
			t.Fatal("stopped worker still accepts connections")
		}
	}
}

func TestWorkerListenFailure(t *testing.T) {
	addr, _ := startDistWorker(t)
	actual, stop, err := RunWorker(addr)
	if err == nil || actual != "" || stop != nil {
		t.Fatalf("occupied address: actual=%q, stop nil=%v, error=%v", actual, stop == nil, err)
	}
}
