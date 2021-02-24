.PHONY: all help empirical simulations plot_empirical plot_simulations suppl_empirical suppl_simulations manuscript

PYTHON ?= python

all: empirical simulations plot_empirical plot_simulations suppl_empirical suppl_simulations manuscript

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  empirical           to run all empirical analyses"
	@echo "  simulations         to run all simulation analyses"
	@echo "  plot_empirical      to generate figures for empirical analyses"
	@echo "  plot_simulations    to generate figures for simulation analyses"
	@echo "  suppl_empirical     to run all supplementary analyses + figure-generating code"
	@echo "  suppl_simulations   to run all supplementary analyses + figure-generating code"
	@echo "  manuscript          to compile a PDF from the manuscript TeX files"
	@echo "  all                 to run *all the things*"
	@echo "  doc                 to create a Jupyter Book of the documentation / walkthrough"

empirical:
	@echo "Running data preprocessing\n"
	$(PYTHON) scripts/empirical/fetch_neurosynth_maps.py
	$(PYTHON) scripts/empirical/fetch_hcp_myelin.py
	@echo "Generating resampling arrays and surrogates\n"
	$(PYTHON) scripts/empirical/get_geodesic_distance.py
	$(PYTHON) scripts/empirical/generate_spin_resamples.py
	$(PYTHON) scripts/empirical/generate_neurosynth_surrogates.py
	$(PYTHON) scripts/empirical/generate_hcp_surrogates.py
	@echo "Running code to analyze NeuroSynth data\n"
	$(PYTHON) scripts/empirical/run_neurosynth_nulls.py
	@echo "Running code to analyze HCP data\n"
	$(PYTHON) scripts/empirical/run_hcp_nulls.py

simulations:
	@echo "Generating simulations and resampling arrays\n"
	$(PYTHON) scripts/empirical/generated_spin_resamples.py
	$(PYTHON) scripts/empirical/get_geodesic_distance.py
	$(PYTHON) scripts/simulations/generate_simulations.py
	@echo "Running correlated simulations"
	$(PYTHON) scripts/simulations/run_simulated_nulls_parallel.py \
		--use_max_knn --spatnull naive-para naive-nonpara vazquez-rodriguez \
								 baum cornblath vasa hungarian moran -- 1000
	$(PYTHON) scripts/simulations/run_simulated_nulls_parallel.py \
		--use_max_knn --spatnull burt2018 burt2020 -- 0 1000
	@echo "Running randomized simulations"
	$(PYTHON) scripts/simulations/run_simulated_nulls_parallel.py --shuffle \
		--use_max_knn --spatnull naive-para naive-nonpara vazquez-rodriguez \
								 baum cornblath vasa hungarian moran -- 1000
	$(PYTHON) scripts/simulations/run_simulated_nulls_parallel.py --shuffle \
		--use_max_knn --spatnull burt2018 burt2020 -- 0 1000
	@echo "Generating Moran's I estimates"
	$(PYTHON) scripts/simulations/run_simulated_nulls_serial.py \
		--use_max_knn --run_moran
	@echo "Combining simulation outputs"
	$(PYTHON) scripts/simulations/combine_moran_outputs.py
	$(PYTHON) scripts/simulations/combine_simnulls_outputs.py

plot_empirical:
	@echo "Running scripts to visualize empirical results\n"
	$(PYTHON) scripts/plot_empirical/viz_perms.py
	$(PYTHON) scripts/plot_empirical/viz_neurosynth_nulls.py
	$(PYTHON) scripts/plot_empirical/viz_hcp_nulls.py
	$(PYTHON) scripts/plot_empirical/viz_hcp_networks.py

plot_simulations:
	@echo "Running scripts to visualize simulation results\n"
	$(PYTHON) scripts/plot_simulations/viz_simulation_examples.py
	$(PYTHON) scripts/plot_simulations/viz_simulation_results.py

suppl_empirical:
	@echo "Running supplementary analyses + generating extra figures\n"
	$(PYTHON) scripts/suppl_empirical/compare_spin_resamples.py
	$(PYTHON) scripts/suppl_empirical/compare_geodesic_travel.py

suppl_simulations:
	@echo "Running supplementary simulation analyses + generating extra figures\n"
	$(PYTHON) scripts/suppl_simulations/compare_comptime.py
	$(PYTHON) scripts/suppl_simulations/compare_nnulls.py

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
