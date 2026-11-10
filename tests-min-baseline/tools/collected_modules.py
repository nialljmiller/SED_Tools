"""pytest plugin (-p collected_modules): print the sed_tools modules imported by the end of collection.

A new test file may import at its top level only modules on this list: importing any other module at
collection moves its module-level lines out of the tests, and the line proof counts them as lost.
"""
import sys


def pytest_collection_finish(session):
    names = sorted(m for m in sys.modules if m == "sed_tools" or m.startswith("sed_tools."))
    print("\nCOLLECTED-MODULES " + " ".join(names))
