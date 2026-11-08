# distributed-training-system-in-golang

Coursework for Rutgers CS 417: a small distributed model-training system
built in two parts. Each project is its own Go module with no third-party
dependencies.

| Directory | Project | What it builds |
| --- | --- | --- |
| [`p1/`](p1/) | Project 1 | Fault-tolerant distributed SGD: a coordinator splits each batch across RPC workers, retries slow or failed workers, and reduces gradients in a fixed order. |
| [`p2/`](p2/) | Project 2 | Primary-backup key-value store used as a replicated parameter server: the primary forwards every write to the backup and waits for its ack, with at-most-once delivery on both servers. |

[`iLab.pdf`](iLab.pdf) covers the iLab machines used for the performance runs.

## Project 1: distributed training

Run the RPC trainer locally using three terminals from `p1/`:

```bash
cd p1

# Terminal 1: first worker
go run . -mode worker -addr 127.0.0.1:9001

# Terminal 2: second worker
go run . -mode worker -addr 127.0.0.1:9002

# Terminal 3: coordinator, splitting each batch into four slices
go run . -mode coord -workers 127.0.0.1:9001,127.0.0.1:9002 -n 4
```

The coordinator prints the final loss. Workers remain running until stopped
with Ctrl+C and can serve another coordinator run. Stopping one worker still
allows training to finish using the remaining worker. Timed-out requests are
retried on another available worker without canceling the original request.

Run the tests, including worker failures and slow replies, with:

```bash
cd p1
go test -race -timeout=60s .
```

## Project 2: primary-backup parameter server

Run a backup, a primary that forwards to it, and a clerk, from `p2/`:

```bash
cd p2

# Terminal 1: backup (no -forward)
go run . -mode kvserver -addr localhost:9002

# Terminal 2: primary, forwarding writes to the backup
go run . -mode kvserver -addr localhost:9001 -forward localhost:9002

# Terminal 3: clerk
go run . -mode clerk -primary localhost:9001 -op put -key x -val 7
go run . -mode clerk -primary localhost:9001 -op get -key x
```

Run the tests with the race detector, and the SGD throughput benchmark:

```bash
cd p2
go test -race .
go test -run '^TestPerf$' -v .   # prints PERF sgd_steps_per_sec=<n>
```

The benchmark runs a 10-dimensional linear-regression training loop (one
`Get` and one `Put` of the model per step) against a primary-backup pair and
fails if the loss does not converge. See [`p2/Report.md`](p2/Report.md) for
the measured number.
