## 📁 2_generators/README.md

### 🔄 Generators

Generators are a type of iterable, like lists or tuples, but instead of storing all values in memory, they yield one item at a time, which makes them ideal for working with large datasets or streams.

In this directory, you will explore:

- **Understanding `yield`**: How `yield` turns a function into a generator and allows pausing/resuming execution.
- **Generator functions vs expressions**: Syntax and use cases of each.
- **Stateful iteration**: Preserving state across multiple calls.
- **Use cases**:
  - Reading large files line by line
  - Lazy data processing pipelines
  - Infinite data streams (e.g., sensor data)

Generators improve performance and reduce memory usage in your Python applications.

**Practice file:** `generator_practice.py`
