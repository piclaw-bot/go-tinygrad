package tensor

import "sync"

// bufferPool reuses allocated buffers to reduce GC pressure.
// Key: buffer size in bytes. Value: pool of []byte slices.
var bufferPool sync.Map // map[int]*sync.Pool

func pooledAlloc(dtype DType, n int) *Buffer {
	size := n * dtype.ByteSize()
	pool, _ := bufferPool.LoadOrStore(size, &sync.Pool{
		New: func() any { return make([]byte, size) },
	})
	data := pool.(*sync.Pool).Get().([]byte)
	// Zero the buffer
	for i := range data {
		data[i] = 0
	}
	return &Buffer{Data: data, DType: dtype, Length: n}
}

func releaseBuffer(b *Buffer) {
	if b == nil || b.Data == nil {
		return
	}
	size := len(b.Data)
	if pool, ok := bufferPool.Load(size); ok {
		pool.(*sync.Pool).Put(b.Data)
	}
}
