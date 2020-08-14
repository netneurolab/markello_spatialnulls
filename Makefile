.PHONY: all help preprocess analysis results visualization supplementary manuscript

PYTHON ?= python

all: preprocess analysis results visualization supplementary manuscript doc

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  preprocess      to preprocess data data for analysis"
	@echo "  analysis        to run the primary computational analyses"
	@echo "  results         to run all results-generating code"
	@echo "  visualization   to generate all figures"
	@echo "  supplementary   to run all supplementary analyses + figure-generating code"
	@echo "  manuscript      to compile a PDF from the manuscript TeX files"
	@echo "  doc             to create a Jupyter Book of the documentation / walkthrough"
	@echo "  all             to run *all the things*"

preprocess:
	@echo "Running data preprocessing\n"
	$(PYTHON) scripts/01_preprocess/fetch_neurosynth_maps.py
	$(PYTHON) scripts/01_preprocess/fetch_hcp_myelin.py

analysis:
	@echo "Running data analyses\n"
	$(PYTHON) scripts/02_analysis/get_geodesic_distance.py
	$(PYTHON) scripts/02_analysis/generate_spin_resamples.py
	$(PYTHON) scripts/02_analysis/generate_neurosynth_surrogates.py
	$(PYTHON) scripts/02_analysis/generate_hcp_surrogates.py

results:
	@echo "Running code to generate results outputs\n"
	$(PYTHON) scripts/03_results/run_hcp_nulls.py
	$(PYTHON) scripts/03_results/run_neurosynth_nulls.py

visualization:
	@echo "Running scripts to generate figures\n"
	$(PYTHON) scripts/04_visualization/viz_perms.py
	$(PYTHON) scripts/04_visualization/viz_neurosynth_nulls.py
	$(PYTHON) scripts/04_visualization/viz_hcp_nulls.py

supplementary:
	@echo "Running supplementary analyses + generating extra figures\n"
	$(PYTHON) scripts/05_supplementary/compare_spin_resamples.py
	$(PYTHON) scripts/05_supplementary/compare_geodesic_travel.py

manuscript:
	@echo "Generating PDF with pdflatex + bibtex"
	@cd manuscript && \
	 rm -f main.pdf && \
	 pdflatex --interaction=nonstopmode main > /dev/null && \
	 bibtex main > /dev/null && \
	 pdflatex --interaction=nonstopmode main > /dev/null && \
	 pdflatex --interaction=nonstopmode main > /dev/null && \
	 rm -f main.aux main.bbl main.blg main.log main.out mainNotes.bib main.synctex.gz
	@echo "Check ./manuscript/main.pdf for generated file"

doc:
	@cd walkthrough && make clean html
