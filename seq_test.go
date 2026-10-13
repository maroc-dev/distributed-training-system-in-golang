package main

import (
	"math"
	"testing"
)

func assertSeqWeightsEqual(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("weight count = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			t.Fatalf("weight[%d] = %.17g, want %.17g", i, got[i], want[i])
		}
	}
}

func TestSeqDeterministic(t *testing.T) {
	cfg := SeqConfig{Seed: 2026, Steps: 300, BatchSize: 64, LR: 0.02}
	assertSeqWeightsEqual(t, RunSeq(cfg), RunSeq(cfg))
}

func TestSeqReducesLoss(t *testing.T) {
	cfg := SeqConfig{Seed: 2026, Steps: 300, BatchSize: 64, LR: 0.02}
	eval := NewBatcher(cfg.Seed, 1024).Next(0)
	initialLoss := Loss(Init(cfg.Seed), eval)
	finalLoss := Loss(RunSeq(cfg), eval)
	if math.IsNaN(finalLoss) || math.IsInf(finalLoss, 0) || finalLoss > initialLoss/50 {
		t.Fatalf("loss = %g, want at most %g (initial loss %g)", finalLoss, initialLoss/50, initialLoss)
	}
	t.Logf("loss: %.6f -> %.6f (%.1fx reduction)", initialLoss, finalLoss, initialLoss/finalLoss)
}

func TestSeqCallbackCopies(t *testing.T) {
	cfg := SeqConfig{Seed: 1, Steps: 5, BatchSize: 32, LR: 0.01}
	var snapshots, saved [][]float64
	cfg.OnStep = func(step int, w []float64) {
		if step != len(snapshots) {
			t.Fatalf("callback step = %d, want %d", step, len(snapshots))
		}
		snapshots = append(snapshots, w)
		saved = append(saved, append([]float64(nil), w...))
	}
	baseline := RunSeq(cfg)
	if len(snapshots) != cfg.Steps {
		t.Fatalf("callback count = %d, want %d", len(snapshots), cfg.Steps)
	}
	for step := range snapshots {
		assertSeqWeightsEqual(t, snapshots[step], saved[step])
	}
	assertSeqWeightsEqual(t, snapshots[cfg.Steps-1], baseline)

	calls := 0
	cfg.OnStep = func(step int, w []float64) {
		if step != calls || step >= len(saved) {
			t.Fatalf("unexpected callback step %d", step)
		}
		assertSeqWeightsEqual(t, w, saved[step])
		for i := range w {
			w[i] = math.NaN()
		}
		calls++
	}
	assertSeqWeightsEqual(t, RunSeq(cfg), baseline)
	if calls != cfg.Steps {
		t.Fatalf("callback count = %d, want %d", calls, cfg.Steps)
	}

	snapshots[cfg.Steps-1][0] = math.NaN()
	assertSeqWeightsEqual(t, baseline, saved[cfg.Steps-1])
}

func TestSeqZeroSteps(t *testing.T) {
	cfg := SeqConfig{Seed: 1, Steps: 0, BatchSize: 32, LR: 0.01}
	cfg.OnStep = func(step int, w []float64) {
		t.Fatalf("unexpected callback at step %d with zero training steps", step)
	}
	assertSeqWeightsEqual(t, RunSeq(cfg), Init(cfg.Seed))
}
