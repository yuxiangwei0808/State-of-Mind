"""The paper's analyses, on all Algonauts test data (Friends S6 and the four Movie10 films) unless noted.
They read the released weights ($STATE_OF_MIND_OUTPUT/spectator-v1/) and benchmark result store
($STATE_OF_MIND_OUTPUT/benchmark-results/) and write outputs under $STATE_OF_MIND_OUTPUT/paper-analysis/.

Run each from the repository root:

    python -m state_of_mind.paper.history_table             Table 2 & App. B.3: main table (7.5, 120, 180 s), paired CIs, r
    python -m state_of_mind.paper.inputs run|report         Table 3: full, without stimulus, without fMRI, another viewer's fMRI
    python -m state_of_mind.paper.collect spectator ...     Exact linear state/stimulus decomposition on held-out recordings
    python -m state_of_mind.paper.decomposition swap-table|state-swap|parcel-benefit|gradshap
                                                         Fig. 3-4, App. D.1-D.2: swaps, cortical gains, GradientSHAP
    python -m state_of_mind.paper.analysis benchmark|collected|isc|maps
                                                         App. B.2 & D.3: horizon curves, updating budgets, eigenmodes, IRF, ISC
    python -m state_of_mind.paper.standard_metrics ...      App. B.5: Parcel r, Global r, and Cls@Top10
    python -m state_of_mind.paper.new_viewer ...            App. B.7: unseen HCP viewer readout calibration (0-15 min)
    python -m state_of_mind.paper.long_horizon run|report   App. C.2: 5-minute open-loop Forecast rollouts
    python -m state_of_mind.paper.generated fit|report      App. E.2: linear forecaster trained on Spectator trajectories
"""
