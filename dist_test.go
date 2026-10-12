package main

import (
	"fmt"
	"math"
	"net"
	"net/rpc"
	"sync"
	"sync/atomic"
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

// These workers deliberately delay or fail RPCs to exercise the scheduler.
type distTestWorker struct {
	grad func(GradArgs, *GradReply) error
}

func (worker *distTestWorker) Grad(args GradArgs, reply *GradReply) error {
	return worker.grad(args, reply)
}

func startDistTestWorker(t *testing.T, grad func(GradArgs, *GradReply) error) string {
	t.Helper()
	server := rpc.NewServer()
	if err := server.RegisterName("Worker", &distTestWorker{grad: grad}); err != nil {
		t.Fatal(err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var finished sync.WaitGroup
	connections := make(map[net.Conn]bool)
	stopped := false
	finished.Add(1)
	go func() {
		defer finished.Done()
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			mu.Lock()
			if stopped {
				mu.Unlock()
				conn.Close()
				return
			}
			connections[conn] = true
			finished.Add(1)
			mu.Unlock()
			go func() {
				defer finished.Done()
				server.ServeConn(conn)
				mu.Lock()
				delete(connections, conn)
				mu.Unlock()
			}()
		}
	}()
	t.Cleanup(func() {
		mu.Lock()
		stopped = true
		listener.Close()
		for conn := range connections {
			conn.Close()
		}
		mu.Unlock()
		finished.Wait()
	})
	return listener.Addr().String()
}

func runDistWithDeadline(t *testing.T, cfg DistConfig) []float64 {
	t.Helper()
	type outcome struct {
		w     []float64
		panic any
	}
	done := make(chan outcome, 1)
	go func() {
		result := outcome{}
		defer func() {
			result.panic = recover()
			done <- result
		}()
		result.w = RunCoordinator(cfg)
	}()
	select {
	case result := <-done:
		if result.panic != nil {
			t.Fatalf("coordinator panicked: %v", result.panic)
		}
		return result.w
	case <-time.After(5 * time.Second):
		t.Fatal("coordinator did not finish")
		return nil
	}
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

func TestCoordinatorCrashAndRefusal(t *testing.T) {
	cfg := DistConfig{Seed: 1, Steps: 20, BatchSize: 32, LR: 0.01,
		NSlices: 4, Timeout: 100 * time.Millisecond}
	var stops []func()
	for i := 0; i < 3; i++ {
		addr, stop := startDistWorker(t)
		cfg.Workers = append(cfg.Workers, addr)
		stops = append(stops, stop)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	unavailable := listener.Addr().String()
	listener.Close()
	cfg.Workers = append([]string{unavailable}, cfg.Workers...)
	cfg.OnStep = func(step int, w []float64) {
		if step == 5 {
			stops[0]()
			stops[1]()
		}
	}
	want := RunLocal(LocalConfig{Seed: cfg.Seed, Steps: cfg.Steps,
		BatchSize: cfg.BatchSize, LR: cfg.LR, NRanks: cfg.NSlices})
	assertSeqWeightsEqual(t, runDistWithDeadline(t, cfg), want)
}

func TestCoordinatorDropsFailedWorker(t *testing.T) {
	for _, malformed := range []bool{false, true} {
		t.Run(fmt.Sprintf("malformed=%v", malformed), func(t *testing.T) {
			var calls atomic.Int32
			bad := startDistTestWorker(t, func(args GradArgs, reply *GradReply) error {
				calls.Add(1)
				if malformed {
					reply.Grad = []float64{1}
					return nil
				}
				return fmt.Errorf("simulated RPC failure")
			})
			good, _ := startDistWorker(t)
			cfg := DistConfig{Seed: 1, Steps: 8, BatchSize: 32, LR: 0.01,
				NSlices: 4, Workers: []string{bad, good}, Timeout: 50 * time.Millisecond}
			want := RunLocal(LocalConfig{Seed: cfg.Seed, Steps: cfg.Steps,
				BatchSize: cfg.BatchSize, LR: cfg.LR, NRanks: cfg.NSlices})
			assertSeqWeightsEqual(t, runDistWithDeadline(t, cfg), want)
			if calls.Load() != 1 {
				t.Fatalf("failed worker called %d times, want exactly once", calls.Load())
			}
		})
	}
}

func TestCoordinatorStraggler(t *testing.T) {
	for _, lateError := range []bool{false, true} {
		t.Run(fmt.Sprintf("lateError=%v", lateError), func(t *testing.T) {
			release := make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			var calls atomic.Int32
			slow := startDistTestWorker(t, func(args GradArgs, reply *GradReply) error {
				if calls.Add(1) == 1 {
					<-release
					if lateError {
						return fmt.Errorf("late failure")
					}
				}
				return (Worker{}).Grad(args, reply)
			})
			t.Cleanup(unblock)
			var duplicates atomic.Int32
			fast := startDistTestWorker(t, func(args GradArgs, reply *GradReply) error {
				if args.Step == 0 && args.SliceIndex == 0 {
					duplicates.Add(1)
				}
				return (Worker{}).Grad(args, reply)
			})
			cfg := DistConfig{Seed: 1, Steps: 20, BatchSize: 32, LR: 0.01,
				NSlices: 1, Workers: []string{slow, fast}, Timeout: 10 * time.Millisecond}
			cfg.OnStep = func(step int, w []float64) {
				if step == 1 {
					unblock() // The old result now races with later training steps.
				}
			}
			want := RunSeq(SeqConfig{Seed: cfg.Seed, Steps: cfg.Steps, BatchSize: cfg.BatchSize, LR: cfg.LR})
			assertSeqWeightsEqual(t, runDistWithDeadline(t, cfg), want)
			if duplicates.Load() != 1 {
				t.Fatalf("duplicate calls = %d, want 1", duplicates.Load())
			}
			if lateError && calls.Load() != 1 {
				t.Fatalf("failed straggler called %d times, want 1", calls.Load())
			}
		})
	}
}

func TestCoordinatorOriginalReplyWins(t *testing.T) {
	originalRelease := make(chan struct{})
	duplicateRelease := make(chan struct{})
	var originalOnce, duplicateOnce sync.Once
	unblockOriginal := func() { originalOnce.Do(func() { close(originalRelease) }) }
	unblockDuplicate := func() { duplicateOnce.Do(func() { close(duplicateRelease) }) }
	var duplicateFinished atomic.Bool
	original := startDistTestWorker(t, func(args GradArgs, reply *GradReply) error {
		<-originalRelease
		return (Worker{}).Grad(args, reply)
	})
	duplicate := startDistTestWorker(t, func(args GradArgs, reply *GradReply) error {
		unblockOriginal()
		<-duplicateRelease
		duplicateFinished.Store(true)
		return (Worker{}).Grad(args, reply)
	})
	t.Cleanup(unblockOriginal)
	t.Cleanup(unblockDuplicate)
	cfg := DistConfig{Seed: 1, Steps: 1, BatchSize: 32, LR: 0.01,
		NSlices: 1, Workers: []string{original, duplicate}, Timeout: 10 * time.Millisecond}
	var originalWon atomic.Bool
	cfg.OnStep = func(step int, w []float64) {
		originalWon.Store(!duplicateFinished.Load())
		unblockDuplicate()
	}
	want := RunSeq(SeqConfig{Seed: cfg.Seed, Steps: cfg.Steps, BatchSize: cfg.BatchSize, LR: cfg.LR})
	assertSeqWeightsEqual(t, runDistWithDeadline(t, cfg), want)
	if !originalWon.Load() {
		t.Fatal("timed-out original was not accepted before the duplicate completed")
	}
}

func TestCoordinatorBusyWorker(t *testing.T) {
	var calls atomic.Int32
	addr := startDistTestWorker(t, func(args GradArgs, reply *GradReply) error {
		if calls.Add(1) == 1 {
			// There is no other worker available when this attempt times out.
			<-time.After(50 * time.Millisecond)
		}
		return (Worker{}).Grad(args, reply)
	})
	cfg := DistConfig{Seed: 1, Steps: 3, BatchSize: 32, LR: 0.01,
		NSlices: 4, Workers: []string{addr}, Timeout: 5 * time.Millisecond}
	want := RunLocal(LocalConfig{Seed: cfg.Seed, Steps: cfg.Steps,
		BatchSize: cfg.BatchSize, LR: cfg.LR, NRanks: cfg.NSlices})
	assertSeqWeightsEqual(t, runDistWithDeadline(t, cfg), want)
	if calls.Load() != int32(cfg.Steps*cfg.NSlices) {
		t.Fatalf("calls = %d, want %d", calls.Load(), cfg.Steps*cfg.NSlices)
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

func TestCoordinatorZeroStepsAndNoSurvivors(t *testing.T) {
	cfg := DistConfig{Seed: 1, Steps: 0, BatchSize: 32, LR: 0.01, NSlices: 4}
	cfg.OnStep = func(step int, w []float64) { t.Fatal("unexpected callback") }
	assertSeqWeightsEqual(t, RunCoordinator(cfg), Init(cfg.Seed))
	bad := startDistTestWorker(t, func(args GradArgs, reply *GradReply) error {
		return fmt.Errorf("no survivor")
	})
	cfg.Steps = 1
	cfg.Workers = []string{bad}
	cfg.Timeout = 50 * time.Millisecond
	defer func() {
		if recover() == nil {
			t.Fatal("expected a clear panic when no workers survive")
		}
	}()
	RunCoordinator(cfg)
}
