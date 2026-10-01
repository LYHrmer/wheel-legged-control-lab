# C31 research evidence subset

The `.pt` file is the C31 event-level lateral-skill checkpoint. It is separate from the frozen B22 rolling policy. This package is a source and selected saved-evidence snapshot, not a portable runnable bundle.

The complete train/evaluation controls, native rows, state arrays and raw trajectories remain in the local archive. `publication_manifest31.json` records their paths, byte counts and SHA-256 hashes; the public subset cannot independently rerun the complete reader. Failed reader attempts and their receipts/logs are included as saved.
