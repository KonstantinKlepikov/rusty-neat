"""
Pytest configuration for rusty_neat_py tests.
"""


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        'markers', 'smoke: quick smoke tests that verify basic functionality'
    )
    config.addinivalue_line(
        'markers', 'slow: slower tests that run full evolution (may take minutes)'
    )
    config.addinivalue_line(
        'markers',
        'integration: integration tests that test multiple components together',
    )
