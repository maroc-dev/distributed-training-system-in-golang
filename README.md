# distributed-training-system-in-golang
A simple, distributed model training system based off Google's MapReduce

Run the RPC trainer locally using three terminals from this directory:

```bash
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
go test -race -timeout=60s .
```
