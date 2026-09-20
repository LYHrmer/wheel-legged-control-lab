# Flat heightfield contact diagnosis

The gpt-6-astra / ultra diagnosis is in `diagnosis/README.md`. All four saved turn runs and the stop snapshot comparisons are retained. Root reran the 9 turn and 28 stop pose comparisons; the two resulting JSON reports are byte-identical to the agent reports. Agent and root each made 74 position/collision queries, with no physical integration. The separate plane preflight adds 37 saved-pose queries and also no physical integration.

Loaded nearly horizontal contact normals occur on the all-zero heightfield; the same poses on a native plane have vertical normals. This establishes a collision-representation difference, not an engine bug or a successful controller. Turn contact force evidence is from actual saved executions; the old stop snapshot comparison is geometry only. The 89–95% leg-motion share is a kinematic identity share, not causal attribution.

Input trajectories remain in the release and turn-limit result folders. The new plane trajectory evidence is in `../heading_flat_plane_01`; old sources and records were not altered.
