/*
perf_test.go -- CS 417/545 Project 2 performance scenario.

DO NOT MODIFY THIS FILE.  Section 2 of the writeup explains what this
measures.  We re-run it on iLab during evaluation.

The workload is a mini distributed-training loop: your parameter
server stores a 10-dimensional weight vector under the key "model".
We run N stochastic-gradient-descent (SGD) steps against it.  Each
step is one Get followed by one Put, exactly as a real parameter
server client (ps-lite, early TensorFlow, PyTorch RPC) would do.

Usage:

    go test -run Perf .

The last line of output must be:

    PERF  sgd_steps_per_sec=<int>

A silent sanity check also fires: if the training loss has not
dropped by at least 50% over the run, the test fails with "loss did
not converge".  That almost always means your primary-backup pair
silently dropped some updates.

If your Clerk is not implemented yet, this test will t.Skip rather
than fail.  Make Parts 1 and 2 pass first, then come back.
*/

package main

import (
	"fmt"
	"math"
	"math/rand"
	"net"
	"net/rpc"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

/* ------------------------------------------------------------------------
   Harness constants.  Fixed so every submission is graded on the same
   workload.  Do not change.
   ------------------------------------------------------------------------ */

const (
	modelDim      = 10      /* dimension of the weight vector            */
	modelKey      = "model" /* single key that stores the whole vector   */
	trainSteps    = 10000   /* measured SGD iterations                   */
	warmupSteps   = 500     /* warm connections and caches before timing */
	learningRate  = 0.01    /* SGD step size                             */
	datasetSize   = 1000    /* synthetic training points                 */
	convergeRatio = 0.5     /* final loss must be at most this * initial */
)

type perfProcess struct {
	addr string
	lis  net.Listener
	kv   *KVServer
}

func startKV(t *testing.T, addr, forward string) *perfProcess {
	t.Helper()
	defer func() { _ = recover() }()
	kv := MakeKVServer(addr, forward)
	srv := rpc.NewServer()
	if err := srv.Register(kv); err != nil {
		t.Skipf("KVServer not registerable: %v", err)
	}
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		t.Skipf("could not listen on %s: %v", addr, err)
	}
	go srv.Accept(lis)
	return &perfProcess{addr: addr, lis: lis, kv: kv}
}

func (p *perfProcess) stop() {
	if p == nil {
		return
	}
	if p.kv != nil {
		p.kv.Kill()
	}
	if p.lis != nil {
		_ = p.lis.Close()
	}
}

/* ------------------------------------------------------------------------
   Linear-regression toy problem.  y = w_true . x + small noise.
   We train a 10-dim w from scratch to match w_true.
   ------------------------------------------------------------------------ */

func makeDataset(rng *rand.Rand) (xs [][]float64, ys []float64, wTrue []float64) {
	wTrue = make([]float64, modelDim)
	for i := range wTrue {
		wTrue[i] = rng.NormFloat64()
	}
	xs = make([][]float64, datasetSize)
	ys = make([]float64, datasetSize)
	for i := 0; i < datasetSize; i++ {
		x := make([]float64, modelDim)
		var dot float64
		for j := range x {
			x[j] = rng.NormFloat64()
			dot += x[j] * wTrue[j]
		}
		xs[i] = x
		ys[i] = dot + 0.01*rng.NormFloat64()
	}
	return
}

func encode(w []float64) string {
	parts := make([]string, len(w))
	for i, v := range w {
		parts[i] = strconv.FormatFloat(v, 'g', -1, 64)
	}
	return strings.Join(parts, ",")
}

func decode(s string) []float64 {
	if s == "" {
		return make([]float64, modelDim)
	}
	parts := strings.Split(s, ",")
	w := make([]float64, len(parts))
	for i, p := range parts {
		f, _ := strconv.ParseFloat(p, 64)
		w[i] = f
	}
	return w
}

func loss(w, wTrue []float64, xs [][]float64, ys []float64) float64 {
	var sum float64
	for i, x := range xs {
		var dot float64
		for j := range x {
			dot += w[j] * x[j]
		}
		d := dot - ys[i]
		sum += d * d
	}
	return sum / float64(len(xs))
}

/* ------------------------------------------------------------------------
   The actual test.
   ------------------------------------------------------------------------ */

func TestPerf(t *testing.T) {
	rng := rand.New(rand.NewSource(1))

	primaryAddr := "127.0.0.1:19001"
	backupAddr := "127.0.0.1:19002"

	backup := startKV(t, backupAddr, "")
	defer backup.stop()
	primary := startKV(t, primaryAddr, backupAddr)
	defer primary.stop()

	time.Sleep(200 * time.Millisecond) /* let listeners settle */

	ck := MakeClerk(primaryAddr)

	xs, ys, wTrue := makeDataset(rng)
	w := make([]float64, modelDim) /* initial weights = 0 */
	safePut(t, ck, modelKey, encode(w))

	initialLoss := loss(w, wTrue, xs, ys)

	/* Warmup. */
	for i := 0; i < warmupSteps; i++ {
		w = sgdStep(t, ck, rng, xs, ys)
	}

	/* Measured phase. */
	start := time.Now()
	for i := 0; i < trainSteps; i++ {
		w = sgdStep(t, ck, rng, xs, ys)
	}
	elapsed := time.Since(start)
	stepsPerSec := int(float64(trainSteps) / elapsed.Seconds())

	finalLoss := loss(w, wTrue, xs, ys)

	fmt.Printf("\nPERF  sgd_steps_per_sec=%d\n", stepsPerSec)
	fmt.Printf("      initial_loss=%.4f  final_loss=%.4f\n",
		initialLoss, finalLoss)

	if math.IsNaN(finalLoss) || finalLoss > initialLoss*convergeRatio {
		t.Fatalf("loss did not converge: initial=%.4f final=%.4f "+
			"(expected final <= %.0f%% of initial).  "+
			"Likely cause: some Put calls were silently dropped "+
			"by the primary-backup pair.",
			initialLoss, finalLoss, convergeRatio*100)
	}
}

func sgdStep(t *testing.T, ck *Clerk, rng *rand.Rand,
	xs [][]float64, ys []float64) []float64 {

	raw := safeGet(t, ck, modelKey)
	w := decode(raw)
	if len(w) != modelDim {
		w = make([]float64, modelDim)
	}

	idx := rng.Intn(len(xs))
	x := xs[idx]
	y := ys[idx]

	var dot float64
	for j := range w {
		dot += w[j] * x[j]
	}
	err := dot - y
	for j := range w {
		w[j] -= learningRate * 2 * err * x[j]
	}

	safePut(t, ck, modelKey, encode(w))
	return w
}

/* ------------------------------------------------------------------------
   Clerk guards: skip the test (not fail) if a method is not implemented.
   ------------------------------------------------------------------------ */

var clerkMu sync.Mutex

func safePut(t *testing.T, ck *Clerk, k, v string) {
	t.Helper()
	clerkMu.Lock()
	defer clerkMu.Unlock()
	defer func() {
		if r := recover(); r != nil {
			t.Skipf("Clerk.Put not implemented yet (%v).  Finish Parts 1-2 first.", r)
		}
	}()
	ck.Put(k, v)
}

func safeGet(t *testing.T, ck *Clerk, k string) string {
	t.Helper()
	clerkMu.Lock()
	defer clerkMu.Unlock()
	var out string
	defer func() {
		if r := recover(); r != nil {
			t.Skipf("Clerk.Get not implemented yet (%v).  Finish Parts 1-2 first.", r)
		}
	}()
	out = ck.Get(k)
	return out
}
