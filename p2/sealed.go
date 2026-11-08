/*
sealed.go -- RPC wire types and shared constants.

DO NOT MODIFY THIS FILE.  Our tests diff your sealed.go against the
reference and refuse to run if it has changed.  Add helper types in
p2.go instead.
*/

package main

/* ========================================================================
   Errors returned in the Err field of every Reply type.
   ======================================================================== */

type Err string

const (
	OK         Err = "OK"
	ErrNoKey   Err = "ErrNoKey"
	ErrNetwork Err = "ErrNetwork"
)

/* ========================================================================
   Operation tags carried in PutAppendArgs and ForwardArgs.
   ======================================================================== */

const (
	OpPut = "Put"
)

/*
Xid is a unique client-operation id for at-most-once delivery.
A retried operation reuses the same Xid; a brand-new operation picks a
fresh one.  See P1 Section 3.3.
*/
type Xid uint64

/* ========================================================================
   Client <-> KVServer RPCs.
   ======================================================================== */

type GetArgs struct {
	Key string
	Xid Xid
}

type GetReply struct {
	Err   Err
	Value string
}

type PutAppendArgs struct {
	Key   string
	Value string
	Op    string /* OpPut */
	Xid   Xid
}

type PutAppendReply struct {
	Err Err
}

/* ========================================================================
   Primary -> Backup replication RPCs.
   ======================================================================== */

/*
ForwardArgs is sent from the primary to the backup for every mutating
client operation that the primary has decided to commit.
*/
type ForwardArgs struct {
	Key   string
	Value string
	Op    string
	Xid   Xid
}

type ForwardReply struct {
	Err Err
}
