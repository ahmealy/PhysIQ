# PhysIQ — Video Demo Narration Script

> **Delivery notes**
> - Speak at a calm, confident pace — not rushed
> - Pause naturally at `[pause]` markers
> - Lines marked `[on screen]` describe what the viewer sees — say these words while that thing is visible
> - Total runtime: ~10 minutes

---

## SCENE 1 — Opening (0:00–0:20)

*[App loading on screen]*

"This is PhysIQ — a platform that replaces slow physics solvers with AI."

"Instead of waiting hours for a simulation, you get results in seconds. And instead of guessing which shape will perform best, you tell the system your target and it finds the design for you."

"Let me walk you through it."

---

## SCENE 2 — Dashboard (0:20–0:50)

*[Dashboard on screen, mouse tracing stat cards]*

"We open on the Dashboard — a live snapshot of the system."

"We can see the model's best validation loss, how many rollouts have been saved, and which simulation domains are available."

*[Mouse on Cylinder Flow domain card]*

"Today we are working with **cylinder flow** — incompressible fluid dynamics around a circular obstacle. A classic CFD benchmark."

"But before we run anything, let's look at the data."

---

## SCENE 3 — Dataset Studio (0:50–2:00)

*[Dataset Studio loading]*

"Dataset Studio gives us full visibility into the training data before we trust any model that was trained on it."

*[Mouse tracing info bar]*

"We have 598 training trajectories. Each one is 600 timesteps — six seconds of simulated flow at ten milliseconds per step."

*[Mouse on velocity distribution chart]*

"This is the distribution of velocity magnitudes at timestep zero — across all trajectories. A clean bell shape. No pathological outliers, consistent physics."

*[Mouse on energy distribution]*

"Same story for kinetic energy. The model saw a well-behaved dataset."

*[Scrolling to Node Type Breakdown]*

"Every node in the mesh has a type. Normal interior nodes, boundary nodes, inflow, outflow. The GNN uses these as features."

*[Mesh Preview visible]*

"And here is an actual mesh from the dataset, colored by velocity. You can already see the wake forming behind the cylinder."

*[Hovering Mesh Quality tooltip]*

"We also compute mesh quality metrics automatically. The shape ratio here tells us how stretched the triangles are — values close to one mean well-formed elements. This matters because highly skewed triangles degrade message passing quality in the graph network."

*[Scrolling to outlier table]*

"Finally, outlier detection. Any trajectory more than three standard deviations from the mean is flagged. Clean dataset — no major outliers here."

---

## SCENE 4 — Train (2:00–3:00)

*[Train page loaded]*

"The training interface. You pick your architecture — Graph Network, Transformer, or GraphSAGE processor — configure your hyperparameters, and start training."

*[Hovering architecture buttons]*

"Each architecture has different inductive biases. GN is the original MeshGraphNets approach. TNS uses attention to let nodes weigh their neighbours differently. SAGE uses neighbourhood aggregation with skip connections."

*[Mouse on Loss Curves area]*

"Training loss and validation loss plot live as training runs. You can spot overfitting early and stop before it matters."

*[Mouse on Remote GPU section]*

"And if you don't have a local GPU — or you want to use a more powerful one — you can point the system at a remote machine over SSH. Training runs there, logs stream back here in real time."

---

## SCENE 5 — Predict (3:00–4:30)

*[Predict page loaded]*

"This is where we run the simulation."

*[Mouse on architecture buttons]*

"The system shows us the best checkpoint for each architecture — epoch number, validation loss. We pick GN."

*[Mouse on trajectory input, tooltip visible]*

"A trajectory is one independent simulation scenario — different cylinder position, different inlet velocity. We have one hundred test trajectories the model has never seen."

"Let's run trajectory three."

*[Clicking Launch Simulation]*

"Launching."

*[Progress bar ticking]*

"The GNN is now running autoregressively — predicting each timestep from the previous one. Six hundred steps."

*[Bar turns green]*

"Done."

*[Scrolling to Performance Metrics]*

"Forty-seven times faster than real time. A traditional CFD solver would take tens of minutes for this. We just did it in seconds."

*[Scrolling to Training Similarity card]*

"This is one of my favourite features. Training Similarity tells us how close this test mesh is to what the model actually trained on — computed in the model's own latent space."

*[Mouse on green badge]*

"Eighty-seven percent. Green light. The model has seen meshes like this before. We can trust this prediction."

"If this number dropped below fifty percent, the model would be extrapolating — and we'd want to verify the result with a full solver."

---

## SCENE 6 — Visualize (4:30–6:30)

*[Visualize page loaded, 3-panel view]*

"Ground truth on the left. GNN prediction in the middle. Error magnitude on the right."

*[Play button clicked, animation running]*

"Watch the velocity field evolve as the flow develops around the cylinder."

*[Animation playing — vortex shedding visible]*

"There it is — vortex shedding. The alternating pressure zones that form behind any bluff body in flow. The GNN has learned to reproduce this entirely from data. No equations hardcoded. No numerical solver."

*[Pause, scrubber dragged]*

"The error stays low throughout. The model tracks the ground truth closely even three hundred timesteps out."

*[DIAGNOSTICS tab clicked]*

"The Diagnostics tab gives us deeper analysis."

*[Overfitting detector visible]*

"The overfitting detector analyses training versus validation loss curves and classifies the training run — healthy, underfitting, or overfitting. This one looks good."

*[PHYSICS tab clicked]*

"And the Physics tab computes derived quantities — vorticity, energy conservation, mass conservation proxy."

*[Pointing at vorticity panels]*

"Vorticity is the curl of the velocity field — a measure of local rotation. Left is ground truth, right is our prediction. The structure matches almost exactly."

*[Energy chart visible]*

"Energy conservation holds throughout the rollout. The model hasn't learned to violate the laws of physics."

---

## SCENE 7 — Generate (6:30–9:00)

*[Generate page loaded]*

"Now for the most powerful feature. Inverse design."

"Instead of asking 'what does this mesh look like?' — we ask 'what mesh will give me this performance?'"

*[Mouse on target slider]*

"I set a target drag proxy value. Lower drag means less resistance — a more aerodynamic design."

*[Sliding to 0.020]*

"Target: 0.020. Now I ask the system: give me four candidate designs that should hit this value."

*[Mouse on Pipeline Steps strip]*

"The pipeline makes this clear. Set a goal. The AI proposes candidates. They get scored instantly with a fast estimator. And any one of them can be sent for a full simulation here."

*[Clicking Generate]*

"Generating."

*[Phase label: Sampling from CVAE...]*

"The CVAE — Conditional Variational Autoencoder — samples from a learned distribution of mesh designs conditioned on our target value."

*[Phase label: Rendering meshes... 1/4]*

"Each candidate renders one by one."

*[Candidates appearing]*

"There they are."

*[Mouse tracing candidate cards]*

"Each card shows the predicted drag value, how confident the model is, the mesh geometry. The green badge means this design is well within the training distribution — the prediction is trustworthy."

*[Clicking best candidate]*

"Let's take a closer look at the best one."

*[Detail panel visible]*

"Predicted drag proxy: 0.019. Very close to our target. Let's verify it."

*[Clicking Analyze]*

"This sends the generated mesh through a full GNN simulation — the same pipeline we just ran manually."

*[Visualize loads with Generated Design badge]*

"And there it is. The generated design, animated. We can verify the drag prediction, look at the flow field, check the physics."

*[Play clicked, animation running]*

"The vortex shedding pattern looks different from the baseline cylinder. The shape change worked."

---

## SCENE 8 — Pipeline & Experiments (9:00–9:45)

*[Pipeline page loaded]*

"The Pipeline view shows the full end-to-end workflow as a directed acyclic graph."

*[Mouse tracing nodes]*

"Data ingestion, preprocessing, graph construction, training, evaluation, prediction, export. Each node shows which files are present, when they were last modified, what config was used."

"This is your audit trail. You always know exactly what state the system is in."

---

## SCENE 9 — Closing (9:45–10:00)

*[Dashboard visible, mouse on PhysIQ logo]*

"PhysIQ. Physics intelligence."

"Fast simulations. Explainable predictions. AI-driven design."

[pause]

"All in the browser."

---

> **Voiceover tips**
> - Record audio separately for clean quality
> - Keep total narration under 9 minutes — the video can have silence during loading moments
> - Re-record any sentence that sounds rushed — slower is always better for technical demos
> - The line *"There it is — vortex shedding"* is the emotional peak of the demo — let it land
