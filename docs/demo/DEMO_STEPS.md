# PhysIQ — Video Demo Steps
### Screen recording guide — follow in order

> **Setup before recording**
> - Resolution: 1920×1080, browser fullscreen
> - Browser zoom: 100%
> - Have a completed rollout already saved (at least 1 pkl in result/)
> - Have training data available (cylinder_flow)
> - Both backend and frontend running
> - Close all other browser tabs
> - Turn off notifications

---

## SCENE 1 — Opening (0:00–0:20)
*Cold open on a blank screen — then navigate to the app*

1. Open browser, type `localhost:5173`, press Enter
2. App loads — **pause 2 seconds** on the Dashboard
3. Slowly move mouse over the **sidebar** — hover each nav item for 1 second (don't click yet)
4. Let eye settle on the **"PhysIQ" logo** in the top-left for 2 seconds

---

## SCENE 2 — Dashboard (0:20–0:50)
*Show the system at a glance*

1. On Dashboard — slowly move mouse over the **4 stat cards** (Training Status, Best Valid Loss, Saved Rollouts, Models Saved)
2. Hover over the **Cylinder Flow** domain card — let the description show
3. Click the **→ arrow** on the Cylinder Flow domain card (navigates to Predict)
4. Immediately click **"Dataset Studio"** in the sidebar — show we are exploring the data first

---

## SCENE 3 — Dataset Studio (0:50–2:00)
*Show the training data quality story*

1. Page loads — **pause 2 seconds** on "Cylinder Flow (CFD)" active tab
2. Slowly scroll down — let the **info bar** (598 trajectories, 600 steps, etc.) come into view
3. Hover slowly over the **Velocity Distribution** chart — tooltip pops up on a bar
4. Move to the **Energy Distribution** chart — hover a bar
5. Scroll down to the **Node Count Distribution** — hover a bar
6. Scroll to **Node Type Breakdown** — mouse traces along the bars (NORMAL → INFLOW → OUTFLOW)
7. Hover over the **Sample Mesh Preview** — it shows the actual mesh colored by velocity
8. Scroll to **Mesh Quality** section
9. Hover the **ⓘ** on "Avg shape ratio" — tooltip appears, **pause 2 seconds** to read
10. Hover the **ⓘ** on "Degenerate faces" — tooltip appears, **pause 2 seconds**
11. Scroll to **Outlier Detection** table — mouse traces the Z-Score column
12. Click **"Flag Simple (Cloth)"** domain button — page updates to cloth data
13. **Pause 3 seconds** — let the cloth stats load
14. Click **"Cylinder Flow (CFD)"** to switch back

---

## SCENE 4 — Train (2:00–3:00)
*Show the training interface — don't actually start training*

1. Click **"Train"** in sidebar
2. Page loads — **pause 2 seconds**
3. Hover over the **GN** architecture button — tooltip appears
4. Hover over the **TNS** architecture button — hover tooltip
5. Hover over the **SAGE** architecture button — hover tooltip
6. Click **"TNS"** to select it — it highlights
7. Hover over the **Learning Rate** field, then the **Epochs** field
8. Scroll down to see the **Loss Curves** area (even if empty/flat)
9. Scroll to **Remote GPU** section — hover the toggle (don't click)
10. Click **"GN"** to switch back to default architecture

---

## SCENE 5 — Predict (3:00–4:30)
*Run a live simulation — this is the main event*

1. Click **"Predict"** in sidebar
2. Page loads — **pause 2 seconds**
3. Mouse moves over the **architecture buttons (GN / TNS / SAGE)** — hover each
4. Click **"GN"** — Champion Model card appears with epoch/loss info
5. Hover over the **ⓘ** next to "Test Trajectory" — tooltip explains what a trajectory is, **pause 3 seconds**
6. Change trajectory index to **3** (click, clear, type 3)
7. Confirm device is **cuda:0** (or CPU if no GPU)
8. Mouse moves to **"Launch Simulation"** button — hover 1 second
9. Click **"Launch Simulation"**
10. **Watch progress bar fill** — don't move mouse, let it run
11. Progress counter ticks (e.g. 50 / 600, 100 / 600...)
12. Bar turns green — "✓ Complete"
13. Scroll down — **Performance Metrics** appear (Elapsed, Steps/sec, Speedup)
14. Continue scrolling — **Training Similarity** card appears
15. Hover over the **green badge** "Model Applicable" — pause 2 seconds
16. Mouse traces the **threshold legend** (≥80% green / 50–80% amber / <50% red)
17. Click **"View Results →"** button

---

## SCENE 6 — Visualize (4:30–6:30)
*The animated physics result*

1. Visualize page loads — **pause 3 seconds** on the 3-panel viewer
2. Mouse points at **"Ground Truth"** label, then **"Prediction"** label, then **"Error Magnitude"** label
3. Click **Play** button (blue circle) — animation starts
4. Let it play for **5–6 seconds** — watch the velocity field evolve
5. Click **Pause**
6. Drag the **scrubber** slowly to t ≈ 300 — then release
7. Drag back to t = 0
8. Click Play again, let run for 3 more seconds, then Pause
9. Mouse points at the **RMSE chart** on the right — hover a point on the line
10. Click the **"DIAGNOSTICS"** tab
11. **Pause 3 seconds** — Overfitting Detector chart loads
12. Mouse traces the colored bands (green = healthy zone)
13. Hover over the **Training Distribution Confidence** score
14. Click the **"PHYSICS"** tab
15. Mouse points at **Vorticity Comparison** panels — "look at that detail"
16. Hover the **Energy Conservation** chart — hover a point on the line
17. **Pause 3 seconds**

---

## SCENE 7 — Generate (6:30–9:00)
*The inverse design — give a target, get candidate shapes*

1. Click **"Generate"** in sidebar
2. Page loads — **pause 2 seconds**
3. Confirm domain is **Cylinder Flow (CFD)**
4. Mouse points at the **Target Drag Proxy slider** — slowly drag it left (lower target value)
5. Set to roughly **0.020** — pause
6. Confirm **Candidates** is set to **4**
7. Confirm **Method** is **"CVAE Sample"**
8. Hover over each **Pipeline Steps** node — point out the flow (Set goal → AI proposes → Quick estimate → Full simulation)
9. Mouse moves to **"Generate"** button — hover 1 second
10. Click **"Generate"**
11. Phase label appears — **"Sampling from CVAE…"** — pause to read
12. Phase label changes — **"Rendering meshes… 1 / 4"** — watch counter tick up
13. Candidates appear one by one — **pause 2 seconds** after each appears
14. All 4 candidates visible — scroll the grid slowly
15. Hover over **candidate 1** — mouse traces the predicted value, OOD badge, mesh thumbnail
16. Hover over **candidate 2** — compare
17. Click the **best candidate** (lowest drag, green badge)
18. **Selected Candidate** detail panel appears on the right — pause 3 seconds
19. Click **"Analyze"** button on the selected candidate
20. Brief spinner — navigates to Visualize with the generated design
21. **Pause 3 seconds** on the result — "Generated Design" badge visible
22. Click Play — let animation run for 5 seconds
23. Pause

---

## SCENE 8 — Pipeline & Experiments (9:00–9:45)
*Quick tour of the remaining pages*

1. Click **"Pipeline & Experiments"** in sidebar
2. Page loads — **pause 2 seconds** on the DAG view
3. Mouse traces the pipeline nodes (Dataset → Preprocess → Train → Evaluate → Predict → Export)
4. Hover a node that is marked done (green) — detail popover appears
5. Scroll down if Experiments section is visible — hover a comparison chart

---

## SCENE 9 — Closing (9:45–10:00)

1. Click **"Dashboard"** in sidebar
2. App returns to the overview
3. Mouse moves to the **PhysIQ logo** in the top-left
4. **Pause 3 seconds** — hold on logo
5. Stop recording

---

> **Post-recording tips**
> - Cut any loading spinners > 3 seconds with a jump cut
> - Add a subtle zoom animation on key moments (tooltips, Training Similarity card, candidate grid)
> - Speed up the rollout progress bar to 2× if it takes > 30 seconds
> - Overlay the narration script (`DEMO_SCRIPT.md`) as voiceover
