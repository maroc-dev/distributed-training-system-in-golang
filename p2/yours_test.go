/*
Your test file.  Nothing here is graded.  Write your own tests as you
go.  Delete or change the examples below.

Useful commands:

    go test .                      run every test in this folder
    go test -run TestSmoke .       run only tests whose name has TestSmoke
    go test -v .                   print each test name as it runs
    go test -race .                catch data races (needed for Part 2)

A good habit: when you find a bug, add a test that would have caught
it.  Then you never chase the same bug twice.
*/

package main

import (
	"fmt"
	"net"
	"net/rpc"
	"reflect"
	"sync"
	"testing"
	"time"
)

/*
A smoke test.  Starts a single KVServer (no backup), writes a key,
reads it back, and checks the value.  Passes once Part 1 is done.
*/
func TestSmokeSingleServer(t *testing.T) {
	addr := "127.0.0.1:21001"

	kv := MakeKVServer(addr, "")
	defer kv.Kill()

	srv := rpc.NewServer()
	if err := srv.Register(kv); err != nil {
		t.Fatalf("register: %v", err)
	}
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer lis.Close()
	go srv.Accept(lis)

	time.Sleep(100 * time.Millisecond)

	ck := MakeClerk(addr)
	ck.Put("hello", "world")
	if got := ck.Get("hello"); got != "world" {
		t.Fatalf("Get(hello) = %q, want %q", got, "world")
	}
}

/*
Template for Part 2.  Uncomment and fill in once ForwardOp works.

func TestSmokePrimaryBackup(t *testing.T) {
    backupAddr  := "127.0.0.1:21011"
    primaryAddr := "127.0.0.1:21012"

    backup := MakeKVServer(backupAddr, "")
    ... register, listen ...

    primary := MakeKVServer(primaryAddr, backupAddr)
    ... register, listen ...

    ck := MakeClerk(primaryAddr)
    ck.Put("k", "v")
    // Reach into the backup directly and check its state matches.
}
*/

type p2TestServer struct {
	kv          *KVServer
	listener    net.Listener
	server      *rpc.Server
	mu          sync.Mutex
	connections map[net.Conn]bool
	finished    sync.WaitGroup
	once        sync.Once
	stopped     bool
}

func startP2TestServer(t *testing.T, backup string) *p2TestServer {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	kv := MakeKVServer(listener.Addr().String(), backup)
	return serveP2TestListener(t, listener, kv, kv)
}

func serveP2TestService(t *testing.T, service any) *p2TestServer {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	return serveP2TestListener(t, listener, service, nil)
}

func serveP2TestListener(t *testing.T, listener net.Listener, service any, kv *KVServer) *p2TestServer {
	t.Helper()
	server := rpc.NewServer()
	if err := server.RegisterName("KVServer", service); err != nil {
		listener.Close()
		t.Fatal(err)
	}
	process := &p2TestServer{kv: kv, listener: listener, server: server, connections: make(map[net.Conn]bool)}
	process.finished.Add(1)
	go process.accept()
	t.Cleanup(process.stop)
	return process
}

func (process *p2TestServer) address() string {
	return process.listener.Addr().String()
}

func (process *p2TestServer) accept() {
	defer process.finished.Done()
	for {
		conn, err := process.listener.Accept()
		if err != nil {
			return
		}
		process.mu.Lock()
		if process.stopped {
			process.mu.Unlock()
			conn.Close()
			return
		}
		process.connections[conn] = true
		process.finished.Add(1)
		process.mu.Unlock()
		go process.serve(conn)
	}
}

func (process *p2TestServer) serve(conn net.Conn) {
	defer process.finished.Done()
	process.server.ServeConn(conn)
	process.mu.Lock()
	delete(process.connections, conn)
	process.mu.Unlock()
}

func (process *p2TestServer) stop() {
	process.once.Do(process.shutdown)
}

func (process *p2TestServer) shutdown() {
	if process.kv != nil {
		process.kv.Kill()
	}
	process.mu.Lock()
	process.stopped = true
	process.listener.Close()
	for conn := range process.connections {
		conn.Close()
	}
	process.mu.Unlock()
	process.finished.Wait()
}

func p2Call(t *testing.T, address, method string, args, reply any) {
	t.Helper()
	var connection rpcConnection
	defer connection.close()
	if err := connection.call(address, "KVServer."+method, args, reply); err != nil {
		t.Fatal(err)
	}
}

func TestP2DuplicateReplies(t *testing.T) {
	process := startP2TestServer(t, "")
	var missing, first, repeated GetReply
	p2Call(t, process.address(), "Get", GetArgs{Key: "key", Xid: 1}, &missing)
	if missing.Err != ErrNoKey || missing.Value != "" {
		t.Fatalf("missing key: %+v", missing)
	}
	var written PutAppendReply
	old := PutAppendArgs{Key: "key", Value: "old", Op: OpPut, Xid: 2}
	p2Call(t, process.address(), "Put", old, &written)
	p2Call(t, process.address(), "Get", GetArgs{Key: "key", Xid: 3}, &first)
	p2Call(t, process.address(), "Put", PutAppendArgs{Key: "key", Value: "new", Op: OpPut, Xid: 4}, &written)
	p2Call(t, process.address(), "Put", old, &written)
	p2Call(t, process.address(), "Get", GetArgs{Key: "key", Xid: 3}, &repeated)
	if repeated != first || first.Value != "old" || written.Err != OK {
		t.Fatalf("duplicate replies changed: first=%+v repeated=%+v put=%+v", first, repeated, written)
	}
	repeated = GetReply{}
	p2Call(t, process.address(), "Get", GetArgs{Key: "key", Xid: 1}, &repeated)
	if repeated != missing {
		t.Fatalf("cached missing-key reply changed: %+v", repeated)
	}
	p2Call(t, process.address(), "Get", GetArgs{Key: "key", Xid: 5}, &repeated)
	if repeated.Err != OK || repeated.Value != "new" {
		t.Fatalf("delayed duplicate overwrote the newer value: %+v", repeated)
	}
}

func TestP2ForwardDuplicate(t *testing.T) {
	process := startP2TestServer(t, "")
	old := ForwardArgs{Key: "key", Value: "old", Op: OpPut, Xid: 1}
	var reply ForwardReply
	p2Call(t, process.address(), "ForwardOp", old, &reply)
	p2Call(t, process.address(), "ForwardOp", ForwardArgs{Key: "key", Value: "new", Op: OpPut, Xid: 2}, &reply)
	p2Call(t, process.address(), "ForwardOp", old, &reply)
	var value GetReply
	p2Call(t, process.address(), "Get", GetArgs{Key: "key", Xid: 3}, &value)
	if reply.Err != OK || value.Value != "new" {
		t.Fatalf("duplicate forward changed state: reply=%+v value=%+v", reply, value)
	}
}

type p2WriteTask struct {
	clerk *Clerk
	index int
	done  chan error
	count int
}

func (task p2WriteTask) run() {
	for i := 0; i < task.count; i++ {
		key := fmt.Sprintf("client-%d/key-%d", task.index, i)
		value := fmt.Sprintf("value-%d-%d", task.index, i)
		task.clerk.Put(key, value)
		task.clerk.Put("shared", value)
		if got := task.clerk.Get(key); got != value {
			task.done <- fmt.Errorf("Get(%s) = %q, want %q", key, got, value)
			return
		}
	}
	task.done <- nil
}

func copyP2Values(kv *KVServer) map[string]string {
	kv.mu.Lock()
	defer kv.mu.Unlock()
	values := make(map[string]string, len(kv.values))
	for key, value := range kv.values {
		values[key] = value
	}
	return values
}

func TestP2ConcurrentReplication(t *testing.T) {
	backup := startP2TestServer(t, "")
	primary := startP2TestServer(t, backup.address())
	done := make(chan error, 4)
	for i := 0; i < 4; i++ {
		clerk := MakeClerk(primary.address())
		t.Cleanup(clerk.connection.close)
		task := p2WriteTask{clerk: clerk, index: i, done: done, count: 30}
		go task.run()
	}
	for i := 0; i < 4; i++ {
		select {
		case err := <-done:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("concurrent clerks did not finish")
		}
	}
	primaryValues, backupValues := copyP2Values(primary.kv), copyP2Values(backup.kv)
	if len(primaryValues) != 121 || !reflect.DeepEqual(primaryValues, backupValues) {
		t.Fatalf("primary and backup differ: primary=%d keys backup=%d keys", len(primaryValues), len(backupValues))
	}
}

type p2RetryService struct {
	kv       *KVServer
	mu       sync.Mutex
	puts     []Xid
	gets     []Xid
	forwards []Xid
}

func (service *p2RetryService) Put(args PutAppendArgs, reply *PutAppendReply) error {
	if err := service.kv.Put(args, reply); err != nil {
		return err
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	service.puts = append(service.puts, args.Xid)
	if len(service.puts) == 1 {
		return fmt.Errorf("simulated lost Put reply")
	}
	return nil
}

func (service *p2RetryService) Get(args GetArgs, reply *GetReply) error {
	if err := service.kv.Get(args, reply); err != nil {
		return err
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	service.gets = append(service.gets, args.Xid)
	if len(service.gets) == 1 {
		var updated PutAppendReply
		service.kv.Put(PutAppendArgs{Key: args.Key, Value: "updated", Op: OpPut, Xid: 100}, &updated)
		return fmt.Errorf("simulated lost Get reply")
	}
	return nil
}

func (service *p2RetryService) ForwardOp(args ForwardArgs, reply *ForwardReply) error {
	if err := service.kv.ForwardOp(args, reply); err != nil {
		return err
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	service.forwards = append(service.forwards, args.Xid)
	if len(service.forwards) == 1 {
		return fmt.Errorf("simulated lost backup acknowledgement")
	}
	return nil
}

func TestP2ClerkRetries(t *testing.T) {
	service := &p2RetryService{kv: MakeKVServer("", "")}
	process := serveP2TestService(t, service)
	clerk := MakeClerk(process.address())
	defer clerk.connection.close()
	clerk.Put("key", "original")
	clerk.Put("other", "value")
	if value := clerk.Get("key"); value != "original" {
		t.Fatalf("retried Get returned %q, want its first reply", value)
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	if len(service.puts) != 3 || service.puts[0] != service.puts[1] || service.puts[1] == service.puts[2] {
		t.Fatalf("incorrect Put retry IDs: %v", service.puts)
	}
	if len(service.gets) != 2 || service.gets[0] != service.gets[1] || service.gets[0] == service.puts[2] {
		t.Fatalf("incorrect Get retry IDs: %v", service.gets)
	}
}

func TestP2ForwardRetry(t *testing.T) {
	service := &p2RetryService{kv: MakeKVServer("", "")}
	backup := serveP2TestService(t, service)
	primary := startP2TestServer(t, backup.address())
	clerk := MakeClerk(primary.address())
	defer clerk.connection.close()
	clerk.Put("key", "value")
	if !reflect.DeepEqual(copyP2Values(primary.kv), copyP2Values(service.kv)) {
		t.Fatal("state differs after retrying backup acknowledgement")
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	if len(service.forwards) != 2 || service.forwards[0] != service.forwards[1] {
		t.Fatalf("forward retry changed Xid: %v", service.forwards)
	}
}

type p2GatedBackup struct {
	kv      *KVServer
	entered chan ForwardArgs
	release chan struct{}
	once    sync.Once
}

func (backup *p2GatedBackup) ForwardOp(args ForwardArgs, reply *ForwardReply) error {
	backup.entered <- args
	<-backup.release
	return backup.kv.ForwardOp(args, reply)
}

func (backup *p2GatedBackup) open() {
	backup.once.Do(backup.unblock)
}

func (backup *p2GatedBackup) unblock() {
	close(backup.release)
}

type p2PutCall struct {
	address string
	args    PutAppendArgs
	reply   PutAppendReply
	err     error
	done    chan struct{}
}

func (call *p2PutCall) run() {
	var connection rpcConnection
	defer connection.close()
	call.err = connection.call(call.address, "KVServer.Put", call.args, &call.reply)
	close(call.done)
}

func TestP2WaitsForBackup(t *testing.T) {
	backup := &p2GatedBackup{kv: MakeKVServer("", ""), entered: make(chan ForwardArgs, 8), release: make(chan struct{})}
	process := serveP2TestService(t, backup)
	primary := startP2TestServer(t, process.address())
	t.Cleanup(backup.open)
	call := &p2PutCall{address: primary.address(), args: PutAppendArgs{Key: "key", Value: "value", Op: OpPut, Xid: 1}, done: make(chan struct{})}
	go call.run()
	select {
	case <-backup.entered:
	case <-time.After(2 * time.Second):
		t.Fatal("primary did not forward the write")
	}
	if copyP2Values(primary.kv)["key"] != "value" || len(copyP2Values(backup.kv)) != 0 {
		t.Fatal("write was not applied locally before waiting for the backup")
	}
	select {
	case <-call.done:
		t.Fatal("primary replied before backup acknowledgement")
	case <-time.After(20 * time.Millisecond):
	}
	backup.open()
	select {
	case <-call.done:
	case <-time.After(2 * time.Second):
		t.Fatal("write did not complete after backup acknowledgement")
	}
	if call.err != nil || call.reply.Err != OK || copyP2Values(backup.kv)["key"] != "value" {
		t.Fatalf("write not replicated before reply: reply=%+v error=%v", call.reply, call.err)
	}
}

func TestP2Kill(t *testing.T) {
	kv := MakeKVServer("", "")
	kv.Kill()
	kv.Kill()
	var read GetReply
	var write PutAppendReply
	var forward ForwardReply
	if err := kv.Get(GetArgs{Key: "key", Xid: 1}, &read); err != nil || read.Err != ErrNetwork {
		t.Fatalf("Get after Kill: %+v error=%v", read, err)
	}
	if err := kv.Put(PutAppendArgs{Key: "key", Xid: 2}, &write); err != nil || write.Err != ErrNetwork {
		t.Fatalf("Put after Kill: %+v error=%v", write, err)
	}
	if err := kv.ForwardOp(ForwardArgs{Key: "key", Xid: 3}, &forward); err != nil || forward.Err != ErrNetwork {
		t.Fatalf("ForwardOp after Kill: %+v error=%v", forward, err)
	}
}

type p2TimeoutBackup struct {
	gate  *p2GatedBackup
	mu    sync.Mutex
	calls []Xid
}

func (backup *p2TimeoutBackup) ForwardOp(args ForwardArgs, reply *ForwardReply) error {
	if err := backup.gate.kv.ForwardOp(args, reply); err != nil {
		return err
	}
	backup.mu.Lock()
	backup.calls = append(backup.calls, args.Xid)
	first := len(backup.calls) == 1
	backup.mu.Unlock()
	if first {
		<-backup.gate.release
	} else {
		backup.gate.open()
	}
	return nil
}

func TestP2NetworkTimeout(t *testing.T) {
	gate := &p2GatedBackup{kv: MakeKVServer("", ""), release: make(chan struct{})}
	backup := &p2TimeoutBackup{gate: gate}
	process := serveP2TestService(t, backup)
	primary := startP2TestServer(t, process.address())
	t.Cleanup(gate.open)
	clerk := MakeClerk(primary.address())
	t.Cleanup(clerk.connection.close)
	done := make(chan error, 1)
	task := p2WriteTask{clerk: clerk, index: 0, count: 1, done: done}
	go task.run()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(4 * time.Second):
		t.Fatal("write did not recover from network timeout")
	}
	if !reflect.DeepEqual(copyP2Values(primary.kv), copyP2Values(gate.kv)) {
		t.Fatal("state differs after timed-out acknowledgement")
	}
	backup.mu.Lock()
	defer backup.mu.Unlock()
	if len(backup.calls) < 3 || backup.calls[0] != backup.calls[1] || backup.calls[len(backup.calls)-1] == backup.calls[0] {
		t.Fatalf("incorrect forwarding IDs after timeout: %v", backup.calls)
	}
}
