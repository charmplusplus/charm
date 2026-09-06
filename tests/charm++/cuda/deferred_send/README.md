Build with `make` against a CUDA runtime with network device communication.
Run `deferred_send` with PEs on two physical GPU hosts, using the same launcher
and GPU mapping as the application's inter-node runs. A one-host launch aborts
instead of silently testing MEMCPY or IPC.

The test sends three device buffers produced on two streams and overwrites its
stack-local host arguments immediately after the proxy returns. It checks that
the slower producer is still unfinished at return, all payloads are correct,
each source callback fires once, another entry method executes, and quiescence
does not arrive before all completion messages. The old blocking sender fails
the unfinished-producer assertion. Two buffers share a stream to exercise
deduplication; the third forces the continuation to join both streams.

Run with both HAPI configurations: event polling and `HAPI_CUDA_CALLBACK`.
Both must print `TEST PASSED`. The delay kernel is warmed up before testing;
avoid heavily oversubscribed hosts that could delay the CPU past the producer.
