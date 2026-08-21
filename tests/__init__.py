"""Test package marker.

Makes helper modules such as ``tests.fake_tokenizer`` importable consistently in
Colab/pytest environments where another top-level ``tests`` package may otherwise
shadow this directory.
"""
