# Optimal-Edge-Fitter

Parameter search for the [fill-simulator](../fill-simulator) strategies. It treats the simulator as a black box: each candidate parameter set is written to a temporary `temp_config_<n>.toml` (copying the latency values from the base config and forcing `use_queue_simulation = true`), the simulator is run on a book-events file with the strategy choice piped to stdin, and `Final P&L`, fill rate and order counts are parsed from its output. An optimizer then proposes the next candidate. Because queue mode is forced, inputs must be `book_events` files.

## What gets tuned

| Strategy | Parameters | Stdin piped to the simulator |
|---|---|---|
| Theo | `place_edge_percent`, `cancel_edge_percent` | `2` |
| Correlation | `place_edge_percent`, `cancel_edge_percent`, `self_weight` | `3`, then the correlation CSV path and the symbol-mapping CSV path |

## Scripts

| Script | Method | Details |
|---|---|---|
| `L-BFGS-B_algorithm.py` | SciPy L-BFGS-B (Theo) | Evaluates a 5x5 grid of `place_edge` (0.005-1.0) and `cancel_ratio` (0.001-0.95), then runs L-BFGS-B from the best 5 grid points with the same bounds, splitting `--iterations` across them. Early evaluations in each block of 20 get random jitter on the parameters, and the first 10 get tiny noise on the fitness. |
| `correlation_LBFGS_algorithm.py` | SciPy L-BFGS-B (Correlation) | The same approach with a third variable, `self_weight` (0.01-0.99), and `place_edge` bounded 0.001-1.0. Takes `--correlation` and `--symbol-mapping`. Results are cached by rounded parameters. |
| `simulated_annealing.py` | Simulated annealing (Theo) | Defaults: 100 iterations, start temperature 100, cooling rate 0.98. Neighbor step sizes shrink with temperature, and when the temperature drops below 10 it is reset to 80 and the search jumps to a random point. |
| `evolutionary_algorithm.py` | Genetic algorithm (Theo) | Defaults: population 20, 10 generations, mutation rate 0.2, crossover rate 0.7. Tournament selection (size 3), five hand-picked starting individuals, Gaussian mutation. |
| `L-BFGS-B_generation.py` | Batch driver | Runs `L-BFGS-B_algorithm.py` for every file matching `--pattern` in a directory. |
| `correlation_LBFGS_generation.py` | Batch driver | The same for the correlation strategy. Press `q` to stop after the current runs finish. |

`cancel_edge` is `place_edge * cancel_ratio` in the L-BFGS-B scripts, which keeps it below `place_edge`.

## Objective

Every script scores a run with a fitness function built from four parts: simulated P&L (weight 3, or 5 in the correlation script), a per-fill term derived from average P&L per filled order, P&L per filled order, and a fill-rate adjustment. Runs that place fewer than 10 orders return an infinite fitness (positive in `L-BFGS-B_algorithm.py`, negative in the other scripts).

- Fill rate is the percentage printed by the simulator (0-100), and the thresholds are compared against that number: a penalty applies below 0.1 (0.05 in the correlation script) and a bonus applies above 0.5 when P&L is positive.
- The per-fill term divides average P&L per fill by a "standard deviation" set to half its absolute value (floored at 0.01), so once at least 5 orders fill it is a constant 20 in magnitude whenever the average is at least $0.02 in size.
- The L-BFGS-B scripts minimize the negated fitness, and the annealing and genetic scripts maximize it.

The scoring is in each script's `calculate_fitness`.

## Requirements

`numpy` and `scipy`, a built `fill_simulator` binary, and book-events files named like `EXCHANGE.book_events.SYMBOL.bin`. The correlation scripts also need an `overall_correlations.csv` and a symbol-mapping CSV.

## Usage

Single symbol:

```bash
python L-BFGS-B_algorithm.py \
  --simulator ../fill-simulator/bin/fill_simulator \
  --config ../fill-simulator/latencies/latency_config_queue.toml \
  --book-events /data/20220801/nasdaq/NASDAQ.book_events.AAPL.bin \
  --output /data/20220801/nasdaq/fillsimulations/theo_strategy.AAPL.bin \
  --iterations 50 --initial-place-edge 0.5 --initial-cancel-edge 0.2
```

Correlation strategy adds `--correlation <overall_correlations.csv>`, `--symbol-mapping <symbol_map.csv>` and `--initial-self-weight`.

The other single-symbol scripts take the same `--simulator --config --book-events --output` arguments. Simulated annealing adds `--iterations --initial-place-edge --initial-cancel-edge`; the genetic algorithm uses `--population --generations --mutation-rate --crossover-rate`.

All symbols in a directory:

```bash
python L-BFGS-B_generation.py \
  --book-events-dir /data/20220801/nasdaq \
  --simulator ../fill-simulator/bin/fill_simulator \
  --output-dir /data/20220801/nasdaq/fillsimulations \
  --max-workers 4 \
  --symbols AAPL MSFT          # optional filter; --pattern defaults to "*.book_events.*.bin"
```

## Output

The two L-BFGS-B scripts append the best parameters and P&L, plus the final point if it differs from the best, to `/home/vir/optimization_results/<SYMBOL>_optimization_results.txt` (`<SYMBOL>_correlation_optimization_results.txt` for the correlation script). Simulated annealing and the genetic algorithm print their results and do not write a results file.
