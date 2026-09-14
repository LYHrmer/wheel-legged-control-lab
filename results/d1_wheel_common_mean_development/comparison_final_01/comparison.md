Full observed episodes; each case has equal weight. Unequal terminal durations remain visible.
These averages do not establish successful completion or terrain traversal.

| Seed | Policy | Completed/cases | vx RMSE | Yaw RMSE | Height RMSE | Return | Durations / reason |
|---|---|---:|---:|---:|---:|---:|---|
| 48001 | zero | 2/2 | 0.042510 | 0.017423 | 0.011676 | 60.641086 | final_right_offset_slow: 32.00s/time_limit; final_diagonal_gentle_turn: 32.00s/time_limit |
| 48001 | unbounded | 0/2 | 0.043521 | 0.036392 | 0.008873 | 48.326321 | final_right_offset_slow: 29.93s/map_boundary; final_diagonal_gentle_turn: 24.02s/map_boundary |
| 48001 | bounded | 0/2 | 0.039743 | 0.044804 | 0.007048 | 45.187309 | final_right_offset_slow: 27.71s/map_boundary; final_diagonal_gentle_turn: 22.53s/map_boundary |
| 48002 | zero | 2/2 | 0.042510 | 0.017423 | 0.011676 | 60.641086 | final_right_offset_slow: 32.00s/time_limit; final_diagonal_gentle_turn: 32.00s/time_limit |
| 48002 | unbounded | 1/2 | 0.196228 | 0.072936 | 0.038974 | 10.556856 | final_right_offset_slow: 8.87s/fall_or_body_contact; final_diagonal_gentle_turn: 32.00s/time_limit |
| 48002 | bounded | 2/2 | 0.123295 | 0.055366 | 0.029376 | 43.389048 | final_right_offset_slow: 32.00s/time_limit; final_diagonal_gentle_turn: 32.00s/time_limit |
| 48003 | zero | 2/2 | 0.042510 | 0.017423 | 0.011676 | 60.641086 | final_right_offset_slow: 32.00s/time_limit; final_diagonal_gentle_turn: 32.00s/time_limit |
| 48003 | unbounded | 0/2 | 0.047161 | 0.046241 | 0.005388 | 50.763506 | final_right_offset_slow: 31.12s/map_boundary; final_diagonal_gentle_turn: 24.82s/map_boundary |
| 48003 | bounded | 0/2 | 0.058641 | 0.054257 | 0.011405 | 43.407902 | final_right_offset_slow: 27.24s/map_boundary; final_diagonal_gentle_turn: 23.68s/map_boundary |

Common prefix per seed/case: recorded transitions 1..K, without padding or continuation.
Any termination penalty on transition K remains included; equal time does not mean equal terrain exposure.

| Seed | Case | K / seconds | Policy | vx RMSE | Yaw RMSE | Height RMSE | Return | Signed x progress |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 48001 | final_right_offset_slow | 2771 / 27.71 | zero | 0.041408 | 0.008128 | 0.011036 | 52.934182 | 5.664495 |
| 48001 | final_right_offset_slow | 2771 / 27.71 | unbounded | 0.042464 | 0.030997 | 0.008966 | 51.956041 | 6.518957 |
| 48001 | final_right_offset_slow | 2771 / 27.71 | bounded | 0.031097 | 0.035400 | 0.006510 | 50.794872 | 5.508715 |
| 48002 | final_right_offset_slow | 887 / 8.87 | zero | 0.035934 | 0.012647 | 0.004927 | 17.337035 | 1.635069 |
| 48002 | final_right_offset_slow | 887 / 8.87 | unbounded | 0.150159 | 0.042914 | 0.030143 | 9.952598 | 2.241417 |
| 48002 | final_right_offset_slow | 887 / 8.87 | bounded | 0.055301 | 0.018930 | 0.023417 | 14.963564 | 1.498809 |
| 48003 | final_right_offset_slow | 2724 / 27.24 | zero | 0.041643 | 0.008150 | 0.010992 | 52.035733 | 5.572316 |
| 48003 | final_right_offset_slow | 2724 / 27.24 | unbounded | 0.044311 | 0.040735 | 0.005722 | 51.724641 | 4.607012 |
| 48003 | final_right_offset_slow | 2724 / 27.24 | bounded | 0.053592 | 0.058230 | 0.009216 | 47.286439 | 3.631743 |
| 48001 | final_diagonal_gentle_turn | 2253 / 22.53 | zero | 0.053733 | 0.028773 | 0.010915 | 42.074132 | 5.686019 |
| 48001 | final_diagonal_gentle_turn | 2253 / 22.53 | unbounded | 0.046260 | 0.043550 | 0.008511 | 41.682693 | 6.266989 |
| 48001 | final_diagonal_gentle_turn | 2253 / 22.53 | bounded | 0.048388 | 0.054209 | 0.007585 | 39.579745 | 5.399521 |
| 48002 | final_diagonal_gentle_turn | 3200 / 32.00 | zero | 0.046370 | 0.024410 | 0.011931 | 60.142674 | 8.095085 |
| 48002 | final_diagonal_gentle_turn | 3200 / 32.00 | unbounded | 0.242297 | 0.102959 | 0.047805 | 11.161113 | 8.286856 |
| 48002 | final_diagonal_gentle_turn | 3200 / 32.00 | bounded | 0.183204 | 0.086130 | 0.033555 | 34.836211 | 6.471820 |
| 48003 | final_diagonal_gentle_turn | 2368 / 23.68 | zero | 0.052441 | 0.028079 | 0.011123 | 44.271291 | 5.988575 |
| 48003 | final_diagonal_gentle_turn | 2368 / 23.68 | unbounded | 0.051815 | 0.052295 | 0.005475 | 44.142612 | 4.844055 |
| 48003 | final_diagonal_gentle_turn | 2368 / 23.68 | bounded | 0.063691 | 0.050284 | 0.013594 | 39.529365 | 3.969137 |

Units: vx m/s; yaw rad/s; height and progress m. Zero is reused, not replicated.
JSON preserves all original full summaries, per-case reward terms, prefix termination flags, and input hashes.
