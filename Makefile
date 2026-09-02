# Default shell execution
SHELL := /bin/bash

# Download marker files
DATA_DIR := data
DOWNLOAD_MARKER := $(DATA_DIR)/.downloads_done

# Target outputs
RUNTIME_CSV := runtime_table.csv
FIGURES_DIR := benchmarks/figures
FIGURES_MARKER := $(FIGURES_DIR)/.figures_done
REPORT := fastrad_scientific_report.md
MULTICASE_CSV := multicase_validation.csv
COHORT_CSV := cohort_throughput.csv
ROBUSTNESS_MARKER := benchmarks/.robustness_done

.PHONY: all download runtime figures report multicase cohort robustness test clean help

# Default target runs everything in priority order (downloads run first)
all: download runtime figures report multicase cohort robustness test

## Data Download Target (Prerequisite)

# Run data preparation scripts before any benchmarks
download: $(DOWNLOAD_MARKER)

$(DOWNLOAD_MARKER): benchmarks/download_rider_pairs.py benchmarks/download_tcia_sample.py
	@mkdir -p $(DATA_DIR)
	uv run python benchmarks/download_rider_pairs.py
	uv run python benchmarks/download_tcia_sample.py
	touch $(DOWNLOAD_MARKER)

## High Priority Targets

# Run runtime performance benchmark -> runtime_table.csv
runtime: $(RUNTIME_CSV)

$(RUNTIME_CSV): benchmarks/run_runtime_performance.py $(DOWNLOAD_MARKER)
	uv run python benchmarks/run_runtime_performance.py

# Generate all 5 PNG/PDF figures in benchmarks/figures/
figures: $(FIGURES_MARKER)

$(FIGURES_MARKER): benchmarks/generate_figures.py $(RUNTIME_CSV)
	@mkdir -p $(FIGURES_DIR)
	uv run python benchmarks/generate_figures.py
	touch $(FIGURES_MARKER)

## Medium Priority Targets

# Regenerate fastrad_scientific_report.md with updated numbers
report: $(REPORT)

$(REPORT): benchmarks/report_generator.py $(RUNTIME_CSV)
	uv run python benchmarks/report_generator.py

# Multi-case parity CSV
multicase: $(MULTICASE_CSV)

$(MULTICASE_CSV): benchmarks/run_multicase_validation.py $(DOWNLOAD_MARKER)
	uv run python benchmarks/run_multicase_validation.py

# Cohort throughput CSV
cohort: $(COHORT_CSV)

$(COHORT_CSV): benchmarks/run_cohort_throughput.py $(DOWNLOAD_MARKER)
	uv run python benchmarks/run_cohort_throughput.py

## Low Priority & Verification Targets

# Verify expanded edge case matrix
robustness: $(ROBUSTNESS_MARKER)

$(ROBUSTNESS_MARKER): benchmarks/run_robustness.py $(DOWNLOAD_MARKER)
	uv run python benchmarks/run_robustness.py
	touch $(ROBUSTNESS_MARKER)

# Run full test suite
test:
	uv run pytest tests/ -v

## Utility Targets

# Clean generated artifacts
clean:
	rm -f $(RUNTIME_CSV) $(REPORT) $(MULTICASE_CSV) $(COHORT_CSV)
	rm -f $(DOWNLOAD_MARKER) $(FIGURES_MARKER) $(ROBUSTNESS_MARKER)
	rm -rf $(FIGURES_DIR)

# Show help menu
help:
	@echo "Available Makefile targets:"
	@echo "  make all        - Download datasets, run all benchmarks, update report, and run tests"
	@echo "  make download   - Run data download scripts (RIDER pairs and TCIA sample)"
	@echo "  make runtime    - Run runtime performance benchmarks"
	@echo "  make figures    - Generate all 5 PNG/PDF figures"
	@echo "  make report     - Regenerate the scientific report"
	@echo "  make multicase  - Run multi-case validation"
	@echo "  make cohort     - Run cohort throughput benchmarks"
	@echo "  make robustness - Run robustness & edge case verification"
	@echo "  make test       - Run full test suite using pytest"
	@echo "  make clean      - Remove generated reports, CSVs, markers, and figures"