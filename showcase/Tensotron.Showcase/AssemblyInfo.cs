using Xunit;

// The Tensotron runtime is a single shared (non-thread-safe) accelerator, so showcase
// tasks must not run concurrently — same constraint as Tensotron.Tests.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
