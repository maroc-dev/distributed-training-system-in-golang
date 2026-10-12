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
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
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
	panic("not implemented yet: Part 1")
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
	panic("not implemented yet: Part 2")
}

/* Part 3 -- Coordinator and workers over RPC (65 points). */

// Worker is the RPC service. You register it under the name "Worker"
// so clients call "Worker.Grad" and "Worker.Ping".
type Worker struct{}

// Grad returns the summed gradient for one slice of a batch.
func (Worker) Grad(args GradArgs, reply *GradReply) error {
	// Write your code here.
	// See project1-desc.pdf, Part 3.1.
	panic("not implemented yet: Part 3.1 (Worker.Grad)")
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
	panic("not implemented yet: Part 3.1 (RunWorker)")
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
	panic("not implemented yet: Part 3 (RunCoordinator)")
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
