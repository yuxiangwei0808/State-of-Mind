"""The paper's analyses, on all Algonauts test data (Friends S6 and the four Movie10 films) unless noted. They read the
benchmark result stores and runs under $NEUROWORLD_OUTPUT (neuroworld/paths.py) and write
$NEUROWORLD_OUTPUT/paper-analysis/ (results-*.json, figs/). docs/reproduce.md maps each table and figure to its script.
Run each from the repository root:

    python -m neuroworld.paper.history_table          main table by history length (7.5, 120, 180 s), paired differences, r
    python -m neuroworld.paper.ablations              ablation table (LaTeX) and its rows split by seconds ahead
    python -m neuroworld.paper.standard_metrics ...   Pearson r, global r and Cls@Top10 of every model (prior work's metrics)
    python -m neuroworld.paper.brainvista 7.5 120 180 BrainVista's rows, its forecasts rescaled by cross-fitted gains
    python -m neuroworld.paper.brainvista_crossfit    BrainVista on the prior-work metrics, and per step at 180 s
    python -m neuroworld.paper.collect ...            saved forecasts on fixed starting points, and Spectator's state/movie split
    python -m neuroworld.paper.analysis benchmark|collected|isc|maps
                                                      skill by seconds ahead and by new frames, per-parcel maps, the state/movie
                                                      split, inter-viewer correlation, cortical maps
    python -m neuroworld.paper.figures [inputs|inside|cortex]
                                                      figures from analysis.py's results; the data helpers of the appendix
                                                      figures drawn in the paper repository (viz/history, viz/properties)
    python -m neuroworld.paper.long_horizon run|report     Forecast up to 5 minutes ahead
    python -m neuroworld.paper.generated fit|report        the linear forecaster trained on Spectator's forecasts
    python -m neuroworld.paper.neuroworld_generated ...    Spectator's forecasts for NeuroWorld's training windows
    python -m neuroworld.paper.new_viewer ...              a new HCP viewer's readout from a few minutes of fMRI
"""
