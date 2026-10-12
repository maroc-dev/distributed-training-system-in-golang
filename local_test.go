package main

import (
	"fmt"
	"math"
	"testing"
)

func TestLocalTraining(t *testing.T) {
	for _, workers := range []int{1, 2, 4, 8} {
		t.Run(fmt.Sprintf("workers=%d", workers), func(t *testing.T) {
			cfg := LocalConfig{Seed: 2026, Steps: 300, BatchSize: 64, LR: 0.02, NRanks: workers}
			w := RunLocal(cfg)
			assertSeqWeightsEqual(t, w, RunLocal(cfg))
			if len(w) != D {
				t.Fatalf("weight count = %d, want %d", len(w), D)
			}
			eval := NewBatcher(cfg.Seed, 1024).Next(0)
			initialLoss := Loss(Init(cfg.Seed), eval)
			finalLoss := Loss(w, eval)
			if math.IsNaN(finalLoss) || math.IsInf(finalLoss, 0) || finalLoss >= initialLoss {
				t.Fatalf("loss = %g, want less than initial loss %g", finalLoss, initialLoss)
			}
			t.Logf("loss: %.6f -> %.6f (%.1fx reduction)", initialLoss, finalLoss, initialLoss/finalLoss)
		})
	}
}

func TestLocalOneWorkerMatchesSeq(t *testing.T) {
	cfg := LocalConfig{Seed: 1, Steps: 260, BatchSize: 32, LR: 0.01, NRanks: 1}
	want := RunSeq(SeqConfig{Seed: cfg.Seed, Steps: cfg.Steps, BatchSize: cfg.BatchSize, LR: cfg.LR})
	assertSeqWeightsEqual(t, RunLocal(cfg), want)
}

func TestLocalOrderedReduction(t *testing.T) {
	configs := []LocalConfig{
		{Seed: 1, Steps: 137, BatchSize: 64, LR: 0.02, NRanks: 4},
		{Seed: -7, Steps: 70, BatchSize: 128, LR: 0.01, NRanks: 8},
		{Seed: 2026, Steps: 5, BatchSize: 8, LR: 0.02, NRanks: 8},
	}
	for _, cfg := range configs {
		t.Run(fmt.Sprintf("seed=%d/batch=%d/workers=%d", cfg.Seed, cfg.BatchSize, cfg.NRanks), func(t *testing.T) {
			// A serial numerical reference checks the prescribed grouping and
			// order independently of the master's channel protocol.
			want := Init(cfg.Seed)
			batcher := NewBatcher(cfg.Seed, cfg.BatchSize)
			size := cfg.BatchSize / cfg.NRanks
			for step := 0; step < cfg.Steps; step++ {
				batch := batcher.Next(step)
				full := make([]float64, D)
				for workerID := 0; workerID < cfg.NRanks; workerID++ {
					start := workerID * size
					partial := Grad(want, batch[start:start+size])
					for i := range full {
						full[i] += partial[i]
					}
				}
				for i := range want {
					want[i] -= cfg.LR * full[i] / float64(cfg.BatchSize)
				}
			}
			assertSeqWeightsEqual(t, RunLocal(cfg), want)
		})
	}
}

func TestLocalZeroSteps(t *testing.T) {
	for _, workers := range []int{1, 4, 8} {
		t.Run(fmt.Sprintf("workers=%d", workers), func(t *testing.T) {
			cfg := LocalConfig{Seed: 1, Steps: 0, BatchSize: 32, LR: 0.01, NRanks: workers}
			assertSeqWeightsEqual(t, RunLocal(cfg), Init(cfg.Seed))
		})
	}
}

func TestLocalInvalidConfig(t *testing.T) {
	configs := []struct {
		name    string
		workers int
		batch   int
	}{
		{"zero workers", 0, 64},
		{"negative workers", -1, 64},
		{"unequal slices", 3, 64},
		{"empty slices", 128, 64},
		{"zero batch", 4, 0},
		{"negative batch", 4, -8},
		{"batch does not divide dataset", 4, 48},
	}
	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected invalid configuration to panic")
				}
			}()
			RunLocal(LocalConfig{Seed: 1, Steps: 1, BatchSize: cfg.batch, LR: 0.01, NRanks: cfg.workers})
		})
	}
}
