## 📁 5_parallel_async/README.md

### ⚙️ Parallel & Asynchronous Processing

Modern applications often require concurrent execution, whether it's handling multiple tasks at once or performing non-blocking I/O operations. This section introduces the foundations of concurrency in Python.

You will learn:

- **Parallelism with multiprocessing**:
  - Using `multiprocessing.Process` for true parallelism
  - Sharing data between processes
  - Avoiding race conditions
- **Threading and concurrent.futures**:
  - Differences between threads and processes
  - Simplifying concurrency with `ThreadPoolExecutor` and `ProcessPoolExecutor`
- **Asynchronous programming with asyncio**:
  - Writing non-blocking code using `async` and `await`
  - Running multiple coroutines concurrently
  - Real-world example: Async HTTP requests or timers

Understanding these tools will help you write responsive, high-performance applications.

**Practice files:**
- `multiprocessing_example.py`
- `threading_vs_multiprocessing.py`
- `asyncio_intro.py`
