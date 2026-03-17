Contributing to fastrad
========================

We welcome community contributions, particularly optimizations mapping novel tensor operations substituting slower scalar implementations!

The PR Process and Gotchas
--------------------------
1. Fork the `fastrad` repository natively.
2. Initialize your local experimental branch: `git checkout -b feature/your-feature`
3. Commit localized logic iterations securely. Ensure standard Python PEP8 formatting using `flake8` or `black`.
4. Run the comprehensive benchmark tests (`python benchmarks/report_generator.py`) guaranteeing PyRadiomics compliance parity before submitting. **Pull requests failing numerical IBSI bounds identical to `PyRadiomics` will not be accepted.**
5. Push your feature branch and initialize a Pull Request.

Submitting Bug Reports
----------------------
When creating issues for extraction exceptions or unexpected behavior, please provide:
* The exact geometric dimensions and modality properties of your `MedicalImage`.
* The `FeatureSettings` initialization blueprint (especially the target `device:` string).
* Standardized terminal outputs or specific PyTorch `OutOfMemoryError` tracebacks.

We do *not* require local patient scan uploads. Synthetic tensor scripts reliably modeling identical data boundaries (using `torch.rand`) are ideal for issue replications securely protecting patient PHI limits.
