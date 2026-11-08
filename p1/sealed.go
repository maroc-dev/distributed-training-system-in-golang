// sealed.go -- the parts of Project 1 you are NOT allowed to modify.
//
// This file contains the reference model, the deterministic data source, and
// the on-wire RPC types. Together they define the fixed inputs that make
// grading bit-for-bit possible. The grading server diffs your submitted
// sealed.go against the reference before running tests; a modified sealed.go
// is a failed submission regardless of what your tests print.
//
// The model is linear regression with mean-squared error. There is no ML in
// the traditional sense: no non-linearities, no stochasticity, no learned
// dropout. Everything is a deterministic function of a seed.
//
// -----------------------------------------------------------------------------
// DO NOT MODIFY THIS FILE. Anything below this line is graded against a
// reference. Read it freely; edit p1.go instead.
// -----------------------------------------------------------------------------
package main

import "math/rand"

// D is the fixed number of features per example and the fixed length of
// every weight vector and gradient.
const D = 16

// Example is one training example.
type Example struct {
	X []float64 // length D
	Y float64
}

// -----------------------------------------------------------------------------
// Sealed model: Init, Loss, Grad. Reduction order is pinned left-to-right.
// -----------------------------------------------------------------------------

// Init returns a fresh weight vector of length D, drawn uniformly from
// [-0.01, +0.01] using a PRNG seeded by `seed`. Deterministic.
func Init(seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	w := make([]float64, D)
	for i := range w {
		w[i] = (r.Float64()*2 - 1) * 0.01
	}
	return w
}

// Loss returns the mean squared error of w over the batch. Sums are taken
// in example order.
func Loss(w []float64, batch []Example) float64 {
	if len(w) != D {
		panic("Loss: len(w) != D")
	}
	if len(batch) == 0 {
		panic("Loss: empty batch")
	}
	var s float64
	for _, e := range batch {
		if len(e.X) != D {
			panic("Loss: example X has wrong length")
		}
		d := dotV(w, e.X) - e.Y
		s += d * d
	}
	return s / float64(len(batch))
}

// Grad returns the SUM (not the mean) of per-example gradients over the
// batch: sum_e [ 2*(dot(w,X)-Y) * X ]. The accumulation is strictly
// left-to-right in example order. This is the pinned reduction order the
// whole project depends on.
//
// Callers that want a mean gradient divide the returned vector by the batch
// size AFTER the sum. Distributed reducers sum per-slice results in slice
// order, then divide by the total batch size.
func Grad(w []float64, batch []Example) []float64 {
	if len(w) != D {
		panic("Grad: len(w) != D")
	}
	if len(batch) == 0 {
		panic("Grad: empty batch")
	}
	g := make([]float64, D)
	for _, e := range batch {
		if len(e.X) != D {
			panic("Grad: example X has wrong length")
		}
		coeff := 2 * (dotV(w, e.X) - e.Y)
		for i := 0; i < D; i++ {
			g[i] += coeff * e.X[i]
		}
	}
	return g
}

func dotV(a, b []float64) float64 {
	var s float64
	for i := 0; i < D; i++ {
		s += a[i] * b[i]
	}
	return s
}

// -----------------------------------------------------------------------------
// Sealed synthetic dataset. Generated on demand from a seed; two calls with
// the same seed produce byte-identical shards.
// -----------------------------------------------------------------------------

const (
	NumShards        = 8
	ExamplesPerShard = 1024
	TotalExamples    = NumShards * ExamplesPerShard
	noiseSigma       = 0.05
)

const (
	wstarKey int64 = 0x574A6CE7DEADBEEF
	shardKey int64 = 0x1E3779B97F4A7C15
	permKey  int64 = 0x3A7C4E45C0FFEE55
)

func wstar(seed int64) []float64 {
	r := rand.New(rand.NewSource(seed ^ wstarKey))
	w := make([]float64, D)
	for i := range w {
		w[i] = r.NormFloat64() * 0.3
	}
	return w
}

// Load returns the examples in shard s, deterministically from `seed`.
func Load(seed int64, s int) []Example {
	if s < 0 || s >= NumShards {
		panic("Load: shard out of range")
	}
	w := wstar(seed)
	r := rand.New(rand.NewSource(seed ^ int64(s+1)*shardKey))
	out := make([]Example, ExamplesPerShard)
	for i := range out {
		x := make([]float64, D)
		for j := range x {
			x[j] = r.NormFloat64()
		}
		var y float64
		for j := range x {
			y += w[j] * x[j]
		}
		y += r.NormFloat64() * noiseSigma
		out[i] = Example{X: x, Y: y}
	}
	return out
}

// LoadAll returns the concatenation of every shard in order.
func LoadAll(seed int64) []Example {
	out := make([]Example, 0, TotalExamples)
	for s := 0; s < NumShards; s++ {
		out = append(out, Load(seed, s)...)
	}
	return out
}

// Batcher yields deterministic batches. Given the same (seed, batchSize),
// two batchers return the same sequence of batches. batchSize must divide
// TotalExamples (currently 8192).
type Batcher struct {
	pool      []Example
	perm      []int
	batchSize int
}

// NewBatcher returns a Batcher over the full seeded dataset.
func NewBatcher(seed int64, batchSize int) *Batcher {
	if batchSize <= 0 || TotalExamples%batchSize != 0 {
		panic("NewBatcher: batchSize must divide TotalExamples")
	}
	b := &Batcher{
		pool:      LoadAll(seed),
		batchSize: batchSize,
	}
	r := rand.New(rand.NewSource(seed ^ permKey))
	b.perm = r.Perm(len(b.pool))
	return b
}

// BatchSize returns the batch size the Batcher was constructed with.
func (b *Batcher) BatchSize() int { return b.batchSize }

// Next returns the batch for training step `step` (0-indexed).
func (b *Batcher) Next(step int) []Example {
	n := len(b.pool)
	start := (step * b.batchSize) % n
	out := make([]Example, b.batchSize)
	for i := 0; i < b.batchSize; i++ {
		out[i] = b.pool[b.perm[start+i]]
	}
	return out
}

// -----------------------------------------------------------------------------
// Sealed RPC wire types. Do not change these; the grader speaks this shape.
// -----------------------------------------------------------------------------

// GradArgs is the request payload for Worker.Grad. Everything the worker
// needs to compute a slice gradient. Because Grad is a pure function of
// these fields, re-issuing after a timeout is always safe.
type GradArgs struct {
	Step       int       // training step, for logging
	SliceIndex int       // which slice this is, for logging
	W          []float64 // current weights (length D)
	Batch      []Example // this slice's examples
}

// GradReply carries the summed slice gradient in example order.
type GradReply struct {
	Grad []float64 // length D
}

// PingArgs is empty; PingReply carries a pong flag.
type PingArgs struct{}
type PingReply struct{ OK bool }
