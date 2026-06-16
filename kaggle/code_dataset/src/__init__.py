# Package marker for the staged Kaggle code dataset copy of `src`.
# Ensures `import src.*` resolves reliably when this folder is on sys.path
# inside a Kaggle kernel. The local repo uses an implicit namespace package
# (no src/__init__.py); this file exists only in the uploaded copy.
