# STEP-08 — DES (EMPTY resets adjacency)

Scenarios evaluated: 373
Runtime: 3.76 seconds

## Settings
```json
{
  "base_paint_time": 1.0,
  "time_per_setup_unit": 1.0,
  "oven_time_per_job": 1.5,
  "empty_time_ratio": 1.0,
  "allow_purge": true,
  "flush_cost_units": 3.0,
  "tie_break_policy": "longer_remaining",
  "run_step05": true,
  "run_step06": true,
  "run_step07": true,
  "f4_weights": null
}
```

## Metrics (head)

method,scenario,solution_id,seed,throughput_jobs,tokens_empty,makespan,pb1_busy_time,pb1_blocked_time,pb1_idle_time,pb2_busy_time,pb2_blocked_time,pb2_idle_time,oven_busy_time,oven_idle_time,pb_block_frac_max,oven_idle_frac,f4
step05,step05_global,,,183,5,283.0,100.0,8192.0,0.0,100.0,8301.0,0.0,274.5,7.5,29.3321554770318,0.026501766784452298,
step06_milp,step06_global,,,183,0,275.5,100.0,3448.5,0.0,100.0,12506.0,0.0,274.5,0.0,45.39382940108893,0.0,
step07_nsga2,step07_global_line::sid0,0,,183,0,275.5,180.0,4447.0,0.0,181.0,4390.5,0.0,274.5,0.0,16.141560798548095,0.0,
step07_nsga2,step07_global_line::sid1,1,,183,0,275.5,180.0,4447.0,0.0,181.0,4390.5,0.0,274.5,0.0,16.141560798548095,0.0,
step07_nsga2,step07_global_line::sid2,2,,183,0,275.5,181.0,4416.5,0.0,179.0,4428.0,0.0,274.5,0.0,16.072595281306715,0.0,
step07_nsga2,step07_global_line::sid3,3,,183,0,275.5,181.0,4416.5,0.0,179.0,4428.0,0.0,274.5,0.0,16.072595281306715,0.0,
step07_nsga2,step07_global_line::sid4,4,,183,0,276.5,201.0,3472.0,0.0,197.0,3924.5,0.0,274.5,0.0,14.193490054249548,0.0,
step07_nsga2,step07_global_line::sid5,5,,183,0,276.5,201.0,3472.0,0.0,197.0,3924.5,0.0,274.5,0.0,14.193490054249548,0.0,
step07_nsga2,step07_global_line::sid6,6,,183,0,276.5,199.0,3474.0,0.0,196.0,3931.5,0.0,274.5,0.0,14.21880650994575,0.0,
step07_nsga2,step07_global_line::sid7,7,,183,0,276.5,199.0,3474.0,0.0,196.0,3931.5,0.0,274.5,0.0,14.21880650994575,0.0,
step07_nsga2,step07_global_line::sid8,8,,183,0,275.5,173.0,4631.5,0.0,185.0,4205.0,0.0,274.5,0.0,16.81125226860254,0.0,
step07_nsga2,step07_global_line::sid9,9,,183,0,275.5,175.0,4505.0,0.0,182.0,4387.5,0.0,274.5,0.0,16.352087114337568,0.0,
