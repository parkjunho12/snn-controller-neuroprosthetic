## 📁 3_context_managers/README.md

### 📦 Context Managers

Context managers simplify resource management by automatically handling setup and teardown operations, like opening and closing files or acquiring and releasing locks.

This directory covers:

- **Using `with` statement**: Ensuring clean resource usage with minimal code.
- **Creating custom context managers**:
  - Implementing `__enter__` and `__exit__` methods
  - Reusable patterns for resource control
- **`contextlib` module**:
  - Using `@contextmanager` decorator
  - Simplifying context manager creation with generators

Mastering context managers leads to safer and more maintainable code, especially when working with external resources.

**Practice file:** `with_statement_custom.py`