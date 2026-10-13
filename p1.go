// p1.go -- the file you edit for Project 1.
//
// The file has three parts, one for each part of the project.
// Read project1-desc.pdf for what each part must do.
//
// Rules:
//   - Do not change the names of the functions in this file.
//   - Do not change the arguments they take or what they return.
//   - Do not change the fields inside SeqConfig, LocalConfig, or
//     DistConfig.
//   - You may add your own helper functions and types anywhere.
//   - Do not modify sealed.go or yours_test.go.
//
// After you finish a part, you can run it by hand:
//   go run . -mode seq
//   go run . -mode local -n 4
//   go run . -mode worker -addr localhost:9001
//   go run . -mode coord  -workers localhost:9001,localhost:9002

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"net/rpc"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
	// You will need more packages as you go. Add them here when you need
	// them. Likely: "encoding/gob", "errors", "net", "net/rpc", "sync",
	// "sync/atomic".
)

/* Part 1 -- Sequential SGD (20 points). See project1-desc.pdf, Part 1. */

// SeqConfig holds the inputs to RunSeq. Do not add or remove fields.
type SeqConfig struct {
	Seed      int64
	Steps     int
	BatchSize int
	LR        float64

	// OnStep may be set by the test code. If it is not nil, call it
	// after every step with a fresh copy of the weights.
	OnStep func(step int, w []float64)
}

// RunSeq trains for cfg.Steps steps and returns the final weights.
func RunSeq(cfg SeqConfig) []float64 {
	// Write your code here.
	// See project1-desc.pdf, Part 1 for what this function must do.
	w := Init(cfg.Seed)
	b := NewBatcher(cfg.Seed, cfg.BatchSize)
	for step := 0; step < cfg.Steps; step++ {
		batch := b.Next(step)
		g := Grad(w, batch)
		for i := range w {
			w[i] -= cfg.LR * g[i] / float64(cfg.BatchSize)
		}
		if cfg.OnStep != nil {
			snapshot := make([]float64, len(w))
			copy(snapshot, w)
			cfg.OnStep(step, snapshot)
		}
	}
	return w
}

/* Part 2 -- Parallel local trainer (15 points). See project1-desc.pdf, Part 2. */

// LocalConfig holds the inputs to RunLocal. NRanks must divide
// BatchSize. Do not add or remove fields.
type LocalConfig struct {
	Seed      int64
	Steps     int
	BatchSize int
	LR        float64
	NRanks    int
}

// RunLocal trains using NRanks goroutines and returns the final weights.
func RunLocal(cfg LocalConfig) []float64 {
	// Write your code here.
	// See project1-desc.pdf, Part 2. Sum partial gradients in rank order so
	// two runs with the same input return the same result, bit for bit.
	if cfg.NRanks <= 0 {
		panic("RunLocal: NRanks must be positive")
	}
	if cfg.BatchSize <= 0 || cfg.BatchSize%cfg.NRanks != 0 {
		panic("RunLocal: NRanks must divide a positive BatchSize")
	}
	batcher := NewBatcher(cfg.Seed, cfg.BatchSize)

	workers := make([]localWorkerChannels, cfg.NRanks)
	for workerID := range workers {
		workers[workerID] = localWorkerChannels{
			batch:   make(chan []Example, 1),
			partial: make(chan []float64, 1),
			total:   make(chan []float64, 1),
		}
	}
	result := make(chan []float64, 1)
	var finished sync.WaitGroup
	finished.Add(cfg.NRanks)
	for workerID := range workers {
		go runLocalWorker(cfg, workerID, batcher, workers, result, &finished)
	}
	finished.Wait()
	return <-result
}

type localWorkerChannels struct {
	batch   chan []Example
	partial chan []float64
	total   chan []float64
}

func runLocalWorker(cfg LocalConfig, workerID int, batcher *Batcher,
	workers []localWorkerChannels, result chan<- []float64, finished *sync.WaitGroup) {
	defer finished.Done()
	const master = 0
	sliceSize := cfg.BatchSize / cfg.NRanks
	w := Init(cfg.Seed)
	for step := 0; step < cfg.Steps; step++ {
		if workerID == master {
			batch := batcher.Next(step)
			for destination := range workers {
				start := destination * sliceSize
				workers[destination].batch <- cloneLocalBatch(batch[start : start+sliceSize])
			}
		}
		batch := <-workers[workerID].batch
		workers[workerID].partial <- Grad(w, batch)

		if workerID == master {
			full := make([]float64, D)
			for source := range workers {
				partial := <-workers[source].partial
				for i := range full {
					full[i] += partial[i]
				}
			}
			for destination := range workers {
				total := make([]float64, D)
				copy(total, full)
				workers[destination].total <- total
			}
		}
		total := <-workers[workerID].total
		for i := range w {
			w[i] -= cfg.LR * total[i] / float64(cfg.BatchSize)
		}
	}
	if workerID == master {
		result <- w
	}
}

func cloneLocalBatch(batch []Example) []Example {
	cloned := make([]Example, len(batch))
	for i, example := range batch {
		x := make([]float64, len(example.X))
		copy(x, example.X)
		cloned[i] = Example{X: x, Y: example.Y}
	}
	return cloned
}

/* Part 3 -- Coordinator and workers over RPC (65 points). */

// Worker is the RPC service. You register it under the name "Worker"
// so clients call "Worker.Grad" and "Worker.Ping".
type Worker struct{}

// Grad returns the summed gradient for one slice of a batch.
func (Worker) Grad(args GradArgs, reply *GradReply) error {
	// Write your code here.
	// See project1-desc.pdf, Part 3.1.
	if len(args.W) != D {
		return fmt.Errorf("Worker.Grad: weight length must be %d", D)
	}
	if len(args.Batch) == 0 {
		return fmt.Errorf("Worker.Grad: batch must not be empty")
	}
	for _, example := range args.Batch {
		if len(example.X) != D {
			return fmt.Errorf("Worker.Grad: feature length must be %d", D)
		}
	}
	if reply == nil {
		return fmt.Errorf("Worker.Grad: reply must not be nil")
	}
	reply.Grad = Grad(args.W, args.Batch)
	return nil
}

// Ping is a health check. Leave it as it is.
func (Worker) Ping(_ PingArgs, reply *PingReply) error {
	reply.OK = true
	return nil
}

// RunWorker starts a worker on addr and returns its actual address
// plus a stop function that shuts it down.
func RunWorker(addr string) (actualAddr string, stop func(), err error) {
	// Write your code here.
	// See project1-desc.pdf, Part 3.1.
	server, err := newWorkerServer(addr, Worker{})
	if err != nil {
		return "", nil, err
	}
	return server.listener.Addr().String(), server.stop, nil
}

type workerServer struct {
	server      *rpc.Server
	listener    net.Listener
	mu          sync.Mutex
	finished    sync.WaitGroup
	connections map[net.Conn]struct{}
	stopped     bool
	once        sync.Once
}

func newWorkerServer(addr string, service any) (*workerServer, error) {
	server := rpc.NewServer()
	if err := server.RegisterName("Worker", service); err != nil {
		return nil, err
	}
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	worker := &workerServer{
		server: server, listener: listener, connections: make(map[net.Conn]struct{}),
	}
	worker.finished.Add(1)
	go worker.acceptConnections()
	return worker, nil
}

func (worker *workerServer) acceptConnections() {
	defer worker.finished.Done()
	for {
		conn, err := worker.listener.Accept()
		if err != nil {
			return
		}
		worker.mu.Lock()
		if worker.stopped {
			worker.mu.Unlock()
			conn.Close()
			return
		}
		worker.connections[conn] = struct{}{}
		worker.finished.Add(1)
		worker.mu.Unlock()
		go worker.serveConnection(conn)
	}
}

func (worker *workerServer) serveConnection(conn net.Conn) {
	defer worker.finished.Done()
	worker.server.ServeConn(conn)
	conn.Close()
	worker.mu.Lock()
	delete(worker.connections, conn)
	worker.mu.Unlock()
}

func (worker *workerServer) stop() {
	worker.once.Do(worker.shutdown)
}

func (worker *workerServer) shutdown() {
	worker.mu.Lock()
	worker.stopped = true
	worker.listener.Close()
	for conn := range worker.connections {
		conn.Close()
	}
	worker.mu.Unlock()
	worker.finished.Wait()
}

// DistConfig holds the inputs to RunCoordinator. Do not add or remove
// fields.
type DistConfig struct {
	Seed      int64
	Steps     int
	BatchSize int
	LR        float64
	NSlices   int
	Workers   []string
	Timeout   time.Duration
	OnStep    func(step int, w []float64)
}

// RunCoordinator drives training end to end and returns the final
// weights.
func RunCoordinator(cfg DistConfig) []float64 {
	// Write your code here.
	// See project1-desc.pdf, Part 3.2 and Part 3.3.
	// You are free to add helper types and helper functions.
	if cfg.NSlices <= 0 || cfg.BatchSize <= 0 || cfg.BatchSize%cfg.NSlices != 0 {
		panic("RunCoordinator: NSlices must divide a positive BatchSize")
	}
	batcher := NewBatcher(cfg.Seed, cfg.BatchSize)
	w := Init(cfg.Seed)
	if cfg.Steps <= 0 {
		return w
	}
	if cfg.Timeout <= 0 {
		panic("RunCoordinator: Timeout must be positive")
	}
	ctx, cancel := context.WithCancel(context.Background())
	master := distMaster{ctx: ctx, cancel: cancel, timeout: cfg.Timeout}
	seen := make(map[string]bool)
	for _, addr := range cfg.Workers {
		addr = strings.TrimSpace(addr)
		if !seen[addr] {
			master.workers = append(master.workers, distWorker{addr: addr})
			seen[addr] = true
		}
	}
	if len(master.workers) == 0 {
		cancel()
		panic("RunCoordinator: provide at least one worker")
	}
	master.results = make(chan distResult, len(master.workers))
	defer master.shutdown()
	for step := 0; step < cfg.Steps; step++ {
		snapshot := append([]float64(nil), w...)
		gradients := master.trainStep(step, snapshot, batcher.Next(step), cfg.NSlices)
		full := make([]float64, D)
		for _, gradient := range gradients {
			for i := range full {
				full[i] += gradient[i]
			}
		}
		for i := range w {
			w[i] -= cfg.LR * full[i] / float64(cfg.BatchSize)
		}
		if cfg.OnStep != nil {
			cfg.OnStep(step, append([]float64(nil), w...))
		}
	}
	return w
}

type distMaster struct {
	ctx        context.Context
	cancel     context.CancelFunc
	timeout    time.Duration
	workers    []distWorker
	results    chan distResult
	finished   sync.WaitGroup
	nextWorker int
}

type distWorker struct {
	addr   string
	dead   bool
	active *distAttempt
}

type distAttempt struct {
	worker   int
	step     int
	slice    int
	deadline time.Time
	expired  bool
}

type distResult struct {
	attempt  *distAttempt
	gradient []float64
	err      error
}

func (master *distMaster) shutdown() {
	master.cancel()
	master.finished.Wait()
}

type workerRPCClient struct {
	client *rpc.Client
}

func (client *workerRPCClient) close() {
	client.client.Close()
}

func (master *distMaster) availableWorker() int {
	for offset := range master.workers {
		workerID := (master.nextWorker + offset) % len(master.workers)
		worker := &master.workers[workerID]
		if !worker.dead && worker.active == nil {
			master.nextWorker = (workerID + 1) % len(master.workers)
			return workerID
		}
	}
	return -1
}

func (master *distMaster) startAttempt(workerID int, args GradArgs) {
	attempt := &distAttempt{
		worker: workerID, step: args.Step, slice: args.SliceIndex,
		deadline: time.Now().Add(master.timeout),
	}
	worker := &master.workers[workerID]
	worker.active = attempt
	addr := worker.addr
	master.finished.Add(1)
	go master.runAttempt(attempt, addr, args)
}

func (master *distMaster) runAttempt(attempt *distAttempt, addr string, args GradArgs) {
	defer master.finished.Done()
	result := distResult{attempt: attempt}
	dialer := net.Dialer{}
	conn, err := dialer.DialContext(master.ctx, "tcp", addr)
	if err == nil {
		client := &workerRPCClient{client: rpc.NewClient(conn)}
		closeOnShutdown := context.AfterFunc(master.ctx, client.close)
		var reply GradReply
		err = client.client.Call("Worker.Grad", args, &reply)
		closeOnShutdown()
		client.close()
		if err == nil && len(reply.Grad) != D {
			err = fmt.Errorf("worker returned %d gradient entries, want %d", len(reply.Grad), D)
		}
		result.gradient = reply.Grad
	}
	result.err = err
	select {
	case master.results <- result:
	case <-master.ctx.Done():
	}
}

func (master *distMaster) trainStep(step int, w []float64, batch []Example, nSlices int) [][]float64 {
	gradients := make([][]float64, nSlices)
	pending := make([]bool, nSlices)
	for i := range pending {
		pending[i] = true
	}
	size := len(batch) / nSlices
	remaining := nSlices
	for remaining > 0 {
		now := time.Now()
		live := 0
		for _, worker := range master.workers {
			if !worker.dead {
				live++
			}
			attempt := worker.active
			if attempt != nil && attempt.step == step && gradients[attempt.slice] == nil &&
				!attempt.expired && !now.Before(attempt.deadline) {
				attempt.expired = true
				pending[attempt.slice] = true
			}
		}
		if live == 0 {
			panic("RunCoordinator: all workers failed before training finished")
		}
		for sliceIndex := range pending {
			if !pending[sliceIndex] || gradients[sliceIndex] != nil {
				continue
			}
			workerID := master.availableWorker()
			if workerID < 0 {
				break
			}
			start := sliceIndex * size
			master.startAttempt(workerID, GradArgs{
				Step: step, SliceIndex: sliceIndex, W: w, Batch: batch[start : start+size],
			})
			pending[sliceIndex] = false
		}

		var deadline time.Time
		for _, worker := range master.workers {
			attempt := worker.active
			if attempt != nil && attempt.step == step && !attempt.expired && gradients[attempt.slice] == nil &&
				(deadline.IsZero() || attempt.deadline.Before(deadline)) {
				deadline = attempt.deadline
			}
		}
		var timer *time.Timer
		var timeout <-chan time.Time
		if !deadline.IsZero() {
			timer = time.NewTimer(time.Until(deadline))
			timeout = timer.C
		}
		select {
		case result := <-master.results:
			attempt := result.attempt
			worker := &master.workers[attempt.worker]
			worker.active = nil
			if result.err != nil {
				worker.dead = true
			}
			if attempt.step == step && gradients[attempt.slice] == nil {
				if result.err != nil {
					pending[attempt.slice] = true
				} else {
					gradients[attempt.slice] = result.gradient
					remaining--
				}
			}
		case <-timeout:
		}
		if timer != nil && !timer.Stop() {
			select {
			case <-timer.C:
			default:
			}
		}
	}
	return gradients
}

/* main -- one binary, four modes. Leave this as it is. */

func main() {
	mode := flag.String("mode", "", "seq | local | worker | coord")
	seed := flag.Int64("seed", 2026, "seed")
	steps := flag.Int("steps", 300, "training steps")
	bs := flag.Int("bs", 64, "batch size")
	lr := flag.Float64("lr", 0.02, "learning rate")
	n := flag.Int("n", 4, "number of ranks (local) or slices (coord)")
	addr := flag.String("addr", "localhost:0", "worker listen address")
	workers := flag.String("workers", "", "comma-separated host:port list (coord)")
	timeout := flag.Duration("timeout", 500*time.Millisecond, "per-attempt timeout")
	flag.Parse()

	switch *mode {
	case "seq":
		eval := NewBatcher(*seed, 1024).Next(0)
		w := RunSeq(SeqConfig{Seed: *seed, Steps: *steps, BatchSize: *bs, LR: *lr})
		fmt.Printf("final loss = %.6f\n", Loss(w, eval))
	case "local":
		eval := NewBatcher(*seed, 1024).Next(0)
		w := RunLocal(LocalConfig{Seed: *seed, Steps: *steps,
			BatchSize: *bs, LR: *lr, NRanks: *n})
		fmt.Printf("final loss = %.6f  (%d ranks)\n", Loss(w, eval), *n)
	case "worker":
		actual, stop, err := RunWorker(*addr)
		if err != nil {
			log.Fatalf("worker start: %v", err)
		}
		fmt.Printf("worker listening on %s\n", actual)
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
		<-sig
		stop()
	case "coord":
		if *workers == "" {
			log.Fatal("coord: provide -workers")
		}
		addrs := strings.Split(*workers, ",")
		eval := NewBatcher(*seed, 1024).Next(0)
		w := RunCoordinator(DistConfig{
			Seed: *seed, Steps: *steps, BatchSize: *bs, LR: *lr,
			NSlices: *n, Workers: addrs, Timeout: *timeout,
		})
		fmt.Printf("final loss = %.6f\n", Loss(w, eval))
	default:
		fmt.Fprintln(os.Stderr, "usage: p1 -mode {seq|local|worker|coord} [flags]")
		flag.PrintDefaults()
		os.Exit(2)
	}
}
