using Xunit;

// Tensotron's TensorRuntime is a process-wide singleton wrapping ONE ILGPU
// Accelerator. Kernel launches + Synchronize on a shared accelerator are not
// safe to interleave across threads, so running test classes in parallel makes
// their buffers race and stomp each other (intermittent zeros/garbage). Serialize
// the whole assembly — correctness over a few seconds of wall-clock.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
