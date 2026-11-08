/*
p2.go -- CS 417/545 Project 2.  Primary-Backup Key-Value Store.

This is the ONLY file you edit (plus yours_test.go for your own testing).
Add any helper types, fields, or package-level variables you need.
Do NOT change the signatures or receiver types of any function that
has an EXPORTED comment above it: those are the public API that our
tests call.
*/

package main

import (
	"bufio"
	"crypto/rand"
	"encoding/binary"
	"encoding/gob"
	"flag"
	"fmt"
	"net"
	"net/rpc"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

/* ========================================================================
   Part 1.  Single-Server Key-Value Service
   ======================================================================== */

/*
KVServer is a single RPC server that stores key -> value.  Part 2
extends it with primary-backup replication by adding a backup
address.
*/
type KVServer struct {
	mu     sync.Mutex
	me     string /* this server's RPC address, e.g. "localhost:9001" */
	backup string /* backup's RPC address, "" if this server is the backup */
	dead   int32  /* set to 1 by Kill */

	/*
	   ADD YOUR CODE HERE: your key-value store, your dedup table,
	   and any helper state you need.
	*/
	writes     sync.Mutex
	values     map[string]string
	getReplies map[getReplyKey]GetReply
	putReplies map[Xid]PutAppendReply
	forward    rpcConnection
}

/*
EXPORTED
MakeKVServer constructs a server.  If backup is "", this server IS
the backup (or a standalone single-server) and will not forward
anything.  If backup is a non-empty address, this server IS the
primary and must forward every mutating op to that address.  The
caller is responsible for calling net/rpc.Register and starting the
listener.
*/
func MakeKVServer(me, backup string) *KVServer {
	/*
	   ADD YOUR CODE HERE.  Typical shape:
	     kv := &KVServer{me: me, backup: backup}
	     ... initialise maps ...
	     return kv
	*/
	return &KVServer{
		me: me, backup: backup,
		values:     make(map[string]string),
		getReplies: make(map[getReplyKey]GetReply),
		putReplies: make(map[Xid]PutAppendReply),
	}
}

/*
EXPORTED
Get returns the current value for args.Key.  Missing keys return "".
Must be at-most-once: a retried call with the same Xid returns the
same reply as the first call.
*/
func (kv *KVServer) Get(args GetArgs, reply *GetReply) error {
	/* ADD YOUR CODE HERE */
	reply.Err = ErrNetwork
	if kv.isDead() {
		return nil
	}
	kv.mu.Lock()
	defer kv.mu.Unlock()
	if kv.isDead() {
		return nil
	}
	cacheKey := getReplyKey{xid: args.Xid, key: args.Key}
	if cached, found := kv.getReplies[cacheKey]; found {
		*reply = cached
		return nil
	}
	value, found := kv.values[args.Key]
	reply.Value = value
	reply.Err = OK
	if !found {
		reply.Err = ErrNoKey
	}
	kv.getReplies[cacheKey] = *reply
	return nil
}

/*
EXPORTED
Put overwrites the value for args.Key.
Must be at-most-once (see Get).
On the primary: must also forward to the backup and wait for its
ack before replying.
*/
func (kv *KVServer) Put(args PutAppendArgs, reply *PutAppendReply) error {
	/* ADD YOUR CODE HERE */
	reply.Err = ErrNetwork
	if kv.isDead() {
		return nil
	}
	kv.writes.Lock()
	defer kv.writes.Unlock()
	defer kv.closeIfDead()
	kv.mu.Lock()
	if kv.isDead() {
		kv.mu.Unlock()
		return nil
	}
	if cached, found := kv.putReplies[args.Xid]; found {
		*reply = cached
		kv.mu.Unlock()
		return nil
	}
	kv.values[args.Key] = args.Value
	kv.mu.Unlock()
	if kv.backup != "" {
		forward := ForwardArgs{Key: args.Key, Value: args.Value, Op: args.Op, Xid: args.Xid}
		for {
			if kv.isDead() {
				return nil
			}
			var acknowledged ForwardReply
			if kv.forward.call(kv.backup, "KVServer.ForwardOp", forward, &acknowledged) == nil && acknowledged.Err == OK {
				break
			}
			if kv.isDead() {
				return nil
			}
			<-time.After(10 * time.Millisecond)
		}
	}
	kv.mu.Lock()
	defer kv.mu.Unlock()
	if !kv.isDead() {
		reply.Err = OK
		kv.putReplies[args.Xid] = *reply
	}
	return nil
}

/* ========================================================================
   Part 2.  Primary-Backup Replication
   ======================================================================== */

/*
EXPORTED
ForwardOp is called by the primary on the backup for every mutating
client operation.  The backup applies the operation (respecting
at-most-once) and returns OK.
*/
func (kv *KVServer) ForwardOp(args ForwardArgs, reply *ForwardReply) error {
	/* ADD YOUR CODE HERE */
	reply.Err = ErrNetwork
	if kv.isDead() {
		return nil
	}
	kv.writes.Lock()
	defer kv.writes.Unlock()
	kv.mu.Lock()
	defer kv.mu.Unlock()
	if kv.isDead() {
		return nil
	}
	if cached, found := kv.putReplies[args.Xid]; found {
		reply.Err = cached.Err
		return nil
	}
	kv.values[args.Key] = args.Value
	kv.putReplies[args.Xid] = PutAppendReply{Err: OK}
	reply.Err = OK
	return nil
}

/* ========================================================================
   Clerk (client-side helper)
   ======================================================================== */

/*
Clerk is the client-side helper type.  It hides retries from the
application.  In P2 the clerk talks to a fixed primary address;
automatic re-routing on failover is scope for the next project.
*/
type Clerk struct {
	mu      sync.Mutex
	primary string /* primary server address */
	xid     uint64 /* atomic counter for fresh Xids */

	/*
	   ADD YOUR CODE HERE: cached RPC client, etc.
	*/
	connection rpcConnection
}

/*
EXPORTED
MakeClerk returns a new Clerk that will send operations to the given
primary address.
*/
func MakeClerk(primary string) *Clerk {
	ck := &Clerk{primary: primary}
	/* ADD YOUR CODE HERE */
	var seed [8]byte
	if _, err := rand.Read(seed[:]); err != nil {
		panic(err)
	}
	ck.xid = binary.LittleEndian.Uint64(seed[:])
	return ck
}

/*
nextXid returns a fresh at-most-once id.  Safe for concurrent use.
Every operation MUST carry a fresh Xid, and a retried operation MUST
reuse it.
*/
func (ck *Clerk) nextXid() Xid {
	return Xid(atomic.AddUint64(&ck.xid, 1))
}

/*
EXPORTED
Get returns the current value for key, or "" if the key is missing.
Keeps retrying through RPC timeouts until it gets a reply.
*/
func (ck *Clerk) Get(key string) string {
	/* ADD YOUR CODE HERE */
	ck.mu.Lock()
	defer ck.mu.Unlock()
	args := GetArgs{Key: key, Xid: ck.nextXid()}
	for {
		var reply GetReply
		if ck.connection.call(ck.primary, "KVServer.Get", args, &reply) == nil {
			if reply.Err == OK || reply.Err == ErrNoKey {
				return reply.Value
			}
		}
		<-time.After(10 * time.Millisecond)
	}
}

/*
EXPORTED
Put overwrites the value for key.  Keeps retrying (see Get).
*/
func (ck *Clerk) Put(key, value string) {
	/* ADD YOUR CODE HERE */
	ck.mu.Lock()
	defer ck.mu.Unlock()
	args := PutAppendArgs{Key: key, Value: value, Op: OpPut, Xid: ck.nextXid()}
	for {
		var reply PutAppendReply
		if ck.connection.call(ck.primary, "KVServer.Put", args, &reply) == nil && reply.Err == OK {
			return
		}
		<-time.After(10 * time.Millisecond)
	}
}

type getReplyKey struct {
	xid Xid
	key string
}

type rpcConnection struct {
	calls   sync.Mutex
	mu      sync.Mutex
	conn    net.Conn
	writer  *bufio.Writer
	encoder *gob.Encoder
	decoder *gob.Decoder
	seq     uint64
}

func (connection *rpcConnection) call(addr, method string, args, reply any) error {
	connection.calls.Lock()
	defer connection.calls.Unlock()
	connection.mu.Lock()
	if connection.conn == nil {
		conn, err := net.DialTimeout("tcp", addr, 500*time.Millisecond)
		if err != nil {
			connection.mu.Unlock()
			return err
		}
		connection.conn = conn
		connection.writer = bufio.NewWriter(conn)
		connection.encoder = gob.NewEncoder(connection.writer)
		connection.decoder = gob.NewDecoder(conn)
	}
	conn, writer, encoder, decoder := connection.conn, connection.writer, connection.encoder, connection.decoder
	connection.mu.Unlock()
	connection.seq++
	var response rpc.Response
	err := conn.SetDeadline(time.Now().Add(500 * time.Millisecond))
	if err == nil {
		err = encoder.Encode(&rpc.Request{ServiceMethod: method, Seq: connection.seq})
	}
	if err == nil {
		err = encoder.Encode(args)
	}
	if err == nil {
		err = writer.Flush()
	}
	if err == nil {
		err = decoder.Decode(&response)
	}
	if err == nil && response.Seq != connection.seq {
		err = fmt.Errorf("rpc: reply sequence %d, want %d", response.Seq, connection.seq)
	}
	if err == nil && response.Error != "" {
		err = rpc.ServerError(response.Error)
	}
	if err == nil {
		err = decoder.Decode(reply)
	}
	if err != nil {
		connection.mu.Lock()
		if connection.conn == conn {
			conn.Close()
			connection.conn = nil
		}
		connection.mu.Unlock()
	}
	return err
}

func (connection *rpcConnection) close() {
	connection.mu.Lock()
	defer connection.mu.Unlock()
	if connection.conn != nil {
		connection.conn.Close()
		connection.conn = nil
	}
}

func (kv *KVServer) closeIfDead() {
	if kv.isDead() {
		kv.forward.close()
	}
}

/* ========================================================================
   Lifecycle helpers.
   ======================================================================== */

/*
EXPORTED
Kill marks the KV server dead.  Running goroutines should notice via
kv.isDead() and exit; subsequent RPCs should return ErrNetwork.
*/
func (kv *KVServer) Kill() {
	atomic.StoreInt32(&kv.dead, 1)
	kv.forward.close()
}

func (kv *KVServer) isDead() bool {
	return atomic.LoadInt32(&kv.dead) != 0
}

/* ========================================================================
   Command-line driver.  Section 5 of the writeup documents the flags.
   ======================================================================== */

func main() {
	mode := flag.String("mode", "", "kvserver | clerk")
	addr := flag.String("addr", "", "kvserver: this server's RPC address, e.g. localhost:9001")
	forward := flag.String("forward", "", "kvserver: backup's RPC address to forward writes to; omit to run as the backup / standalone")
	primary := flag.String("primary", "", "clerk: primary server's RPC address")
	op := flag.String("op", "", "clerk: get | put")
	key := flag.String("key", "", "clerk: key")
	val := flag.String("val", "", "clerk: value for put")
	flag.Parse()

	switch *mode {
	case "kvserver":
		if *addr == "" {
			die("kvserver mode needs -addr")
		}
		kv := MakeKVServer(*addr, *forward)
		rpc.Register(kv)
		serveForever(*addr)
	case "clerk":
		if *primary == "" || *op == "" || *key == "" {
			die("clerk mode needs -primary -op -key")
		}
		ck := MakeClerk(*primary)
		switch *op {
		case "get":
			fmt.Println(ck.Get(*key))
		case "put":
			ck.Put(*key, *val)
		default:
			die("unknown -op: " + *op)
		}
	default:
		die("set -mode to kvserver or clerk")
	}
}

func die(msg string) {
	fmt.Fprintln(os.Stderr, msg)
	os.Exit(2)
}

func serveForever(addr string) {
	/*
	   ADD YOUR CODE HERE: net.Listen on tcp addr, loop over rpc.Accept.
	*/
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		die(fmt.Sprintf("listen: %v", err))
	}
	defer listener.Close()
	rpc.Accept(listener)
}
