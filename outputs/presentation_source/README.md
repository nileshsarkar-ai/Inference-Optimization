# Presentation source

`build_review.mjs` produces the current 18-slide **Inference Optimization** presentation. It reads the saved measurements directly and creates nine editable experiment charts containing all 360 capture observations and all 36 scheduling metric observations, with separate means. It also contains original paper figures, linked references and the research question. `presenter_notes.json` supplies the native speaking notes for all 18 slides.

The visible order is: cover and research question (1–2); previous literature divider and SOTA evidence (3–7); our experiments divider, system and protocols (8–11); our results divider, all measured graph panels and numeric interpretation (12–16); one future comparison (17); references (18).

The source template files are stored in `template_assets.zip`. This keeps only the final presentation as a visible PPTX in the local codebase. Extract the source assets into a temporary build directory when rebuilding.

## Rebuild on the original Mac

Run from `/Users/nileshsarkar/Documents/SaturateLLM/` using the bundled presentation runtime. Adjust these runtime paths on another machine. The experiment data and citations remain under `outputs/`.

```bash
mkdir -p work/presentation-assets
python3 -m zipfile -e outputs/presentation_source/template_assets.zip work/presentation-assets
ln -s /Users/nileshsarkar/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules node_modules
INFERENCE_WORKSPACE="$PWD" TEMPLATE_ASSET_DIR="$PWD/work/presentation-assets" RUNTIME_NODE_MODULES=/Users/nileshsarkar/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules /Users/nileshsarkar/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node outputs/presentation_source/build_review.mjs rebuild
```

Skip the symlink command if the local `node_modules` link already exists. `node_modules/` and `work/` are ignored by Git. The builder imports the presentation finalizer from the installed skill path near its top; that path and the Python runtime must exist or be adapted deliberately.

Use a new suffix for each build. Rebuilt files and renders go into `work/` so they do not create competing final presentations in the repository. After inspecting a revised deck, replace the single root-level `Inference_Optimization_Final.pptx` with that verified version.

The final deck uses the selected Simple Light Mode design, with Helvetica Neue and native charts/tables. Rendered checks use Artifact Tool. Import into Apple Keynote was also checked, including native presenter notes. Microsoft PowerPoint and a physical HDMI/projector setup were not tested. See `../Keynote_Presenter_Setup.md` for the classroom display check.

`write_results_report.py` regenerates `outputs/Results.md` from the recorded analysis. It does not run GPU experiments or change GPU lifecycle state.
