# Tests

Normal tests are the default suite and must run without Raspberry Pi boards, cameras, STM32, UART/RS485 hardware, a display server, or a PyQt UI session.

Install the project in editable development mode and run the normal suite with:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

When adding coverage, choose the test owner by behavior/responsibility rather than by production filename. First search for an existing test that already owns the invariant or a nearby boundary. Prefer extending an existing case, adding a parameterized case, or reusing an existing fixture/helper before creating another test file.

Use a new test file only when the behavior has a distinct responsibility or failure mode that does not fit an existing owner. Production-code changes alone are not a reason to create a new file.

Future hardware-dependent tests must be clearly separated from the normal suite and must not be required for the default `pytest` run.
