package main

import (
	"fmt"
	"math"
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

func TestCoordinatorHealthy(t *testing.T) {
	for _, counts := range []struct{ workers, slices int }{{1, 4}, {2, 1}, {2, 4}, {4, 8}} {
		t.Run(fmt.Sprintf("workers=%d/slices=%d", counts.workers, counts.slices), func(t *testing.T) {
			cfg := DistConfig{Seed: 2026, Steps: 300, BatchSize: 64, LR: 0.02,
				NSlices: counts.slices, Timeout: 500 * time.Millisecond}
			for i := 0; i < counts.workers; i++ {
				addr, _ := startDistWorker(t)
				cfg.Workers = append(cfg.Workers, addr)
			}
			w := RunCoordinator(cfg)
			assertSeqWeightsEqual(t, w, RunCoordinator(cfg))
			want := RunLocal(LocalConfig{Seed: cfg.Seed, Steps: cfg.Steps,
				BatchSize: cfg.BatchSize, LR: cfg.LR, NRanks: cfg.NSlices})
			assertSeqWeightsEqual(t, w, want)
			eval := NewBatcher(cfg.Seed, 1024).Next(0)
			initial, final := Loss(Init(cfg.Seed), eval), Loss(w, eval)
			if math.IsNaN(final) || math.IsInf(final, 0) || final > initial/20 {
				t.Fatalf("loss = %g, want at most %g", final, initial/20)
			}
			t.Logf("loss: %.6f -> %.6f (%.1fx reduction)", initial, final, initial/final)
		})
	}
}

func TestCoordinatorCallbacks(t *testing.T) {
	addr, _ := startDistWorker(t)
	cfg := DistConfig{Seed: 1, Steps: 6, BatchSize: 32, LR: 0.01,
		NSlices: 4, Workers: []string{addr}, Timeout: 100 * time.Millisecond}
	var snapshots, saved [][]float64
	cfg.OnStep = func(step int, w []float64) {
		if step != len(snapshots) {
			t.Fatalf("step = %d, want %d", step, len(snapshots))
		}
		snapshots = append(snapshots, w)
		saved = append(saved, append([]float64(nil), w...))
	}
	baseline := RunCoordinator(cfg)
	if len(snapshots) != cfg.Steps {
		t.Fatalf("callbacks = %d, want %d", len(snapshots), cfg.Steps)
	}
	for step := range saved {
		assertSeqWeightsEqual(t, snapshots[step], saved[step])
	}
	assertSeqWeightsEqual(t, baseline, saved[cfg.Steps-1])
	calls := 0
	cfg.OnStep = func(step int, w []float64) {
		if step != calls || step >= cfg.Steps {
			t.Fatalf("unexpected callback step %d", step)
		}
		assertSeqWeightsEqual(t, w, saved[step])
		for i := range w {
			w[i] = math.NaN()
		}
		calls++
	}
	assertSeqWeightsEqual(t, RunCoordinator(cfg), baseline)
	if calls != cfg.Steps {
		t.Fatalf("callbacks = %d, want %d", calls, cfg.Steps)
	}
	snapshots[cfg.Steps-1][0] = math.NaN()
	assertSeqWeightsEqual(t, baseline, saved[cfg.Steps-1])
}
