"""JWST instrument-selection tool: VULCAN-JAX chemistry -> ExoJax RT -> Pandeia noise.

A PandExo-style planning GUI: pick a science goal (detect molecule X on a
WASP-39b-like planet), and the tool runs the live VULCAN-JAX + ExoJax forward
model locally, simulates each JWST instrument mode's transit-depth precision
with the real STScI Pandeia ETC engine, and ranks the modes by detection
significance.

Entry point:  streamlit run jwst_tool/app.py   (from the vulcan_exojax_run dir)
"""
