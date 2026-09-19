.PHONY: test verify reference-checks quality corpus audit site inventory reviewer-check \
	reproduce-small reproduce-pilot reproduce-p001-r1 verify-p001 reproduce-flagship \
	analyze-pilot paper working-preprint release-r001 verify-r001 verify-datasets setup

setup:
	python -m pip install -e ".[dev]"

test:
	python -m ruff check .
	python -m pytest

reference-checks:
	python scripts/run_reference_checks.py

verify-datasets:
	python scripts/build_dataset_cards.py
	python scripts/verify_dataset.py --dataset sklearn_digits
	python scripts/verify_dataset.py --dataset fashion_mnist

quality: test reference-checks verify-datasets
	python scripts/audit_data_leakage.py
	python scripts/build_experiment_ledger.py
	python scripts/evaluate_phase3_quality.py

corpus: quality
	python scripts/build_benchmark_manifest.py
	python scripts/build_corpus_manifests.py
	python scripts/build_dataset_packages.py
	python scripts/build_experiment_database.py
	python scripts/build_model_manifest.py
	python scripts/build_research_status.py
	python scripts/build_research_catalogs.py
	python visualizations/scripts/build_f001_framework.py
	python scripts/validate_visualizations.py
	python scripts/build_repository_inventory.py
	python scripts/evaluate_release_maturity.py
	python scripts/build_research_site.py
	python scripts/build_synthesis.py
	python scripts/build_presentation.py
	python scripts/validate_synthesis.py
	python scripts/evaluate_release_maturity.py
	python scripts/build_release_manifest.py
	python scripts/validate_release.py
	unlearn-bench validate
	unlearn-bench status

inventory:
	python scripts/build_repository_inventory.py
	python scripts/evaluate_release_maturity.py

site: corpus

reviewer-check: test reference-checks
	python scripts/build_benchmark_manifest.py
	unlearn-bench validate
	python scripts/validate_visualizations.py
	python scripts/validate_release.py
	unlearn-bench evidence

audit: corpus
	python scripts/validate_claims.py
	python scripts/check_paper_ready.py

verify: quality

reproduce-small:
	python scripts/run_experiment.py --config configs/experiments/digits_linear_tiny.yaml

reproduce-pilot:
	python scripts/run_experiment.py --config configs/experiments/fashion_cnn_random_01pct_pilot.yaml

verify-p001:
	python scripts/verify_p001_archive.py

reproduce-p001-r1: verify-p001
	python scripts/run_reference_checks.py
	python scripts/validate_visualizations.py
	@echo "P001 R1 artifact reproduction and independent metric validation: PASS"

reproduce-flagship:
	@echo "Blocked: no confirmatory flagship manifest has been frozen."
	@exit 1

analyze-pilot:
	python scripts/analyze_counterfactual_pilot.py

paper:
	python scripts/build_publication_artifacts.py
	python scripts/build_preprint_values.py
	python scripts/estimate_equivalence_v2_1_compute.py
	python scripts/check_paper_ready.py
	latexmk -pdf -halt-on-error -cd paper/manuscript.tex
	latexmk -pdf -halt-on-error -cd paper/supplementary/supplement.tex

working-preprint:
	python scripts/build_preprint_values.py
	latexmk -pdf -halt-on-error -cd paper/main.tex

release-r001: paper
	python scripts/build_r001_release.py

verify-r001:
	python scripts/verify_r001_release.py --release release/R001
