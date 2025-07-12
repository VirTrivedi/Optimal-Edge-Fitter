#!/usr/bin/env python3
"""
Parameter optimizer for TheoStrategy using L-BFGS-B algorithm.
This script finds optimal placeEdgePercent and cancelEdgePercent parameters.
"""

import subprocess
import re
import random
import argparse
import os
import time
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple

# Configuration
DEFAULT_CONFIG_PATH = "/home/vir/fill-simulator/latencies/latency_config_queue.toml"
DEFAULT_BOOK_EVENTS_PATH = "/data/20220801/nasdaq/NASDAQ.book_events.AAPL.bin"
DEFAULT_OUTPUT_PATH = "/data/20220801/nasdaq/fillsimulations/theo_strategy.AAPL.bin"
MAX_ITERATIONS = 50  # Maximum number of function evaluations


class LBFGSOptimizer:
    """
    Optimizer for TheoStrategy parameters using L-BFGS-B algorithm.
    """
    def __init__(
        self, 
        simulator_path: str, 
        config_path: str, 
        book_events_path: str,
        output_path: str,
        initial_place_edge: float = 0.5,
        initial_cancel_edge: float = 0.2
    ):
        """
        Initialize the optimizer with paths and initial parameters.
        """
        self.simulator_path = simulator_path
        self.config_path = config_path
        self.book_events_path = book_events_path
        self.output_path = output_path
        self.strategy_md_latency_ns = 1000
        self.exchange_latency_ns = 10000
        self.extract_base_config_values()
        self.initial_place_edge = initial_place_edge
        self.initial_cancel_edge = min(initial_cancel_edge, initial_place_edge * 0.95)
        self.best_place_edge = initial_place_edge
        self.best_cancel_edge = initial_cancel_edge
        self.best_pnl = float('-inf')
        self.best_fitness = float('-inf')
        self.history: List[Dict] = []
        self.eval_count = 0
        self.evaluation_cache = {}

    def extract_base_config_values(self):
        """
        Extract static values from base config.
        """
        try:
            with open(self.config_path, 'r') as f:
                content = f.read()
                
            # Extract latency values with regex
            md_match = re.search(r'strategy_md_latency_ns\s*=\s*(\d+)', content)
            if md_match:
                self.strategy_md_latency_ns = int(md_match.group(1))
                
            ex_match = re.search(r'exchange_latency_ns\s*=\s*(\d+)', content)
            if ex_match:
                self.exchange_latency_ns = int(ex_match.group(1))
        except Exception as e:
            print(f"Warning: Could not extract base config values: {e}")
    
    def run_simulation(self, place_edge: float, cancel_edge: float) -> Dict:
        """
        Run the fill simulator with specified parameters and return results.
        """
        # Check if we've already evaluated this parameter combination
        cache_key = (round(place_edge, 4), round(cancel_edge, 4))
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        # Create temporary config file with the parameters
        temp_config_path = self.create_temp_config(place_edge, cancel_edge)
        
        # Prepare command to run simulator with TheoStrategy
        cmd = [
            self.simulator_path,
            self.book_events_path,
            self.output_path,
            temp_config_path
        ]
        
        # Run simulator and capture output
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(input="2\n", timeout=60)
            
            if process.returncode != 0:
                print(f"Error running simulator: {stderr}")
                results = {
                    'pnl': float('-inf'),
                    'fill_rate': 0.0,
                    'orders_placed': 0,
                    'orders_filled': 0
                }
            else:
                # Parse results
                results = self.parse_simulation_results(stdout)
            
            # Clean up temp config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            
            # Cache the results
            self.evaluation_cache[cache_key] = results
            return results
            
        except Exception as e:
            print(f"Exception running simulation: {e}")
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            return {
                'pnl': float('-inf'),
                'fill_rate': 0.0,
                'orders_placed': 0,
                'orders_filled': 0
            }
    
    def create_temp_config(self, place_edge: float, cancel_edge: float) -> str:
        """
        Create a temporary config file with the parameters.
        """
        temp_config_path = f"temp_config_{random.randint(1000, 9999)}.toml"
        
        with open(temp_config_path, 'w') as f:
            f.write(f"""
[latency]
strategy_md_latency_ns = {self.strategy_md_latency_ns}
exchange_latency_ns = {self.exchange_latency_ns}

[simulation]
use_queue_simulation = true

[strategy]
place_edge_percent = {place_edge}
cancel_edge_percent = {cancel_edge}
""")
        
        return temp_config_path
    
    def parse_simulation_results(self, output: str) -> Dict:
        """
        Parse simulation results from the output.
        """
        results = {
            'pnl': 0.0,
            'fill_rate': 0.0,
            'orders_placed': 0,
            'orders_filled': 0
        }
        
        # Extract P&L
        pnl_match = re.search(r"Final P&L: \$([+-]?\d+(\.\d+)?)", output)
        if pnl_match:
            results['pnl'] = float(pnl_match.group(1))
        
        # Extract fill rate
        fill_rate_match = re.search(r"Fill Rate: ([+-]?\d+(\.\d+)?)\%", output)
        if fill_rate_match:
            results['fill_rate'] = float(fill_rate_match.group(1))
        
        # Extract orders placed and filled
        orders_placed_match = re.search(r"Total Orders Placed: (\d+)", output)
        if orders_placed_match:
            results['orders_placed'] = int(orders_placed_match.group(1))
            
        orders_filled_match = re.search(r"Total Orders Filled: (\d+)", output)
        if orders_filled_match:
            results['orders_filled'] = int(orders_filled_match.group(1))
        
        return results
    
    def calculate_fitness(self, results: Dict) -> float:
        """
        Calculate fitness score for simulation results with risk adjustment.
        Returns negative fitness for minimization.
        """
        pnl = results['pnl']
        fill_rate = results['fill_rate']
        orders_placed = results['orders_placed']
        orders_filled = results.get('orders_filled', 0)
        
        if orders_placed < 10:
            return float('inf')
        
        # Calculate sharpe-like ratio if we have enough filled orders
        if orders_filled >= 5:
            # Estimate average P&L per trade
            avg_pnl_per_trade = pnl / orders_filled if orders_filled > 0 else 0
            
            # Use a default standard deviation
            default_std_dev = abs(avg_pnl_per_trade) * 0.5
            std_dev = max(0.01, default_std_dev)
            
            # Calculate risk-adjusted return component
            risk_adjusted_return = avg_pnl_per_trade / std_dev
            risk_component = risk_adjusted_return * 10
        else:
            risk_component = 0
        
        # Fill rate components
        if fill_rate < 0.1:
            fill_rate_penalty = 100
        else:
            fill_rate_penalty = 0
        
        # Bonus for higher fill rates when P&L is positive
        fill_rate_bonus = 0
        if pnl > 0 and fill_rate > 0.5:
            fill_rate_bonus = fill_rate * 5
        
        # Reward higher P&L per filled order
        pnl_efficiency = 0
        if orders_filled > 0:
            pnl_per_filled = pnl / orders_filled
            if pnl > 0:
                pnl_efficiency = pnl_per_filled * 10.0
            else:
                pnl_efficiency = pnl_per_filled * 5.0
        
        # Calculate fitness (negative for minimization)
        fitness = -1 * ((pnl * 3.0) + (risk_component * 5.0) + pnl_efficiency - fill_rate_penalty + fill_rate_bonus)
        
        return fitness
    
    def parameter_transformation(self, params: np.ndarray) -> Tuple[float, float]:
        """Transform parameters to respect constraints while encouraging exploration."""
        # Base transformation
        place_edge = max(0.005, min(1.0, params[0]))
        cancel_ratio = min(max(params[1], 0.001), 0.95)
        
        # For early iterations in each optimization run, add randomness
        local_count = self.eval_count % 20
        
        if local_count < 5:
            # Scale randomness by iteration
            rand_scale = 0.2 * (1.0 - local_count/5.0)
            place_edge = max(0.005, min(1.0, place_edge + random.uniform(-rand_scale, rand_scale)))
            cancel_ratio = min(max(cancel_ratio + random.uniform(-rand_scale/2, rand_scale/2), 0.001), 0.95)
        
        # Calculate final cancel_edge
        cancel_edge = place_edge * cancel_ratio
        
        return place_edge, cancel_edge
    
    def objective_function(self, params: np.ndarray) -> float:
        """
        Objective function for L-BFGS-B.
        Takes a parameter vector and returns a scalar fitness to minimize.
        """
        # Transform parameters to respect constraints
        place_edge, cancel_edge = self.parameter_transformation(params)
        
        # Ensure place_edge is reasonable
        if place_edge < 0.005 or place_edge > 1.0:
            return float('inf')
        
        # Run simulation
        results = self.run_simulation(place_edge, cancel_edge)
        
        # Calculate fitness
        fitness = self.calculate_fitness(results)
        
        # Add small noise to prevent premature convergence due to identical values
        if self.eval_count < 10:
            noise = random.uniform(-0.001, 0.001)
            fitness += noise

        # Track evaluation for logging
        self.eval_count += 1
        
        # Update best solution if needed
        if -fitness > self.best_fitness:
            self.best_fitness = -fitness
            self.best_pnl = results['pnl']
            self.best_place_edge = place_edge
            self.best_cancel_edge = cancel_edge
        
        # Add to history
        self.history.append({
            'iteration': self.eval_count,
            'place_edge': place_edge,
            'cancel_edge': cancel_edge,
            'pnl': results['pnl'],
            'fill_rate': results['fill_rate'],
            'fitness': -fitness
        })
        
        # Print progress
        print(f"Iteration {self.eval_count}: place_edge={place_edge:.4f}, "
              f"cancel_edge={cancel_edge:.4f}, P&L=${results['pnl']:.2f}, "
              f"Fill Rate={results['fill_rate']:.2f}%, Fitness={-fitness:.2f}")
        
        return fitness
    
    def callback(self, xk: np.ndarray, **kwargs):
        """
        Callback function for L-BFGS-B to track progress.
        """
        if self.eval_count % 10 == 0:
            print(f"\n----- BEST RESULTS SO FAR (ITERATION {self.eval_count}) -----")
            print(f"Best Place Edge: {self.best_place_edge:.4f}%")
            print(f"Best Cancel Edge: {self.best_cancel_edge:.4f}%")
            print(f"Best P&L: ${self.best_pnl:.2f}")
            print(f"Best Fitness: {self.best_fitness:.2f}")
            print("--------------------------------------------\n")
    
    def grid_search(self, place_steps: int = 5, cancel_ratio_steps: int = 5) -> List[np.ndarray]:
        """
        Perform a grid search to find promising starting points.
        """
        print("\n--- Running grid search to find promising starting points ---")
        
        # Define grid points
        place_values = np.linspace(0.005, 1.0, place_steps)
        cancel_ratios = np.linspace(0.001, 0.95, cancel_ratio_steps)
        
        grid_results = []
        
        # Evaluate all grid points
        for place_edge in place_values:
            for cancel_ratio in cancel_ratios:
                cancel_edge = place_edge * cancel_ratio
                
                # Run simulation
                results = self.run_simulation(place_edge, cancel_edge)
                fitness = self.calculate_fitness(results)
                
                # Track evaluation in history
                self.eval_count += 1
                self.history.append({
                    'iteration': self.eval_count,
                    'place_edge': place_edge,
                    'cancel_edge': cancel_edge,
                    'pnl': results['pnl'],
                    'fill_rate': results['fill_rate'],
                    'fitness': -fitness
                })
                
                # Update best solution if needed
                if -fitness > self.best_fitness:
                    self.best_fitness = -fitness
                    self.best_pnl = results['pnl']
                    self.best_place_edge = place_edge
                    self.best_cancel_edge = cancel_edge                
                
                # Store results
                grid_results.append((fitness, place_edge, cancel_ratio))
                
                # Print progress
                print(f"Grid point {self.eval_count}: place_edge={place_edge:.4f}, "
                    f"cancel_edge={cancel_edge:.4f}, P&L=${results['pnl']:.2f}, "
                    f"Fill Rate={results['fill_rate']:.2f}%, Fitness={-fitness:.2f}")
        
        # Sort by fitness
        grid_results.sort(key=lambda x: x[0])
        
        # Return the 5 best parameter sets
        best_params = []
        for _, place_edge, cancel_ratio in grid_results[:5]:
            best_params.append(np.array([place_edge, cancel_ratio]))
        
        print("\n--- Grid search complete. Using top 5 points for L-BFGS-B optimization ---")
        return best_params

    def run_optimization(self, max_iterations: int = MAX_ITERATIONS) -> Dict:
        """
        Run the L-BFGS-B optimization process with starting points from grid search.
        """
        print("Starting L-BFGS-B optimization...")
        
        # Try multiple starting points from grid search
        best_result = None
        best_fitness = float('-inf')
        
        # Reset tracking between optimization runs
        self.best_pnl = float('-inf')
        self.best_fitness = float('-inf')
        self.best_place_edge = self.initial_place_edge
        self.best_cancel_edge = self.initial_cancel_edge
        
        # Get promising starting points from grid search
        starting_points = self.grid_search(5, 5)
        
        # Add initial parameters as a starting point
        initial_point = np.array([
            self.initial_place_edge,
            self.initial_cancel_edge / self.initial_place_edge
        ])
        
        # If grid search didn't find any good points, use random + initial
        if not starting_points:
            starting_points = [initial_point]
            for _ in range(4):
                random_place = random.uniform(0.005, 1.0)
                random_cancel_ratio = random.uniform(0.001, 0.95)
                starting_points.append(np.array([random_place, random_cancel_ratio]))
        
        # Run L-BFGS-B from each starting point
        for i, start_params in enumerate(starting_points):
            print(f"\n--- Starting L-BFGS-B optimization run #{i+1} ---")
            print(f"Starting point: place_edge={start_params[0]:.4f}, cancel_ratio={start_params[1]:.4f}")
            
            # Run optimization from this starting point
            result = minimize(
                self.objective_function,
                start_params,
                method='L-BFGS-B',
                bounds=[(0.005, 1.0), (0.001, 0.95)],
                callback=self.callback,
                options={
                    'maxiter': max(10, max_iterations // len(starting_points)),
                    'ftol': 1e-3,
                    'gtol': 1e-3,
                    'disp': True
                }
            )
            
            # Always keep the first result as a fallback
            if best_result is None:
                best_result = result
                best_fitness = self.best_pnl
            
            # Check if this run produced better results
            if self.best_pnl > best_fitness:
                best_result = result
                best_fitness = self.best_pnl
        
        # Get optimized parameters
        place_edge = self.best_place_edge
        cancel_edge = self.best_cancel_edge
        
        # Final evaluation to ensure we have correct values
        final_results = self.run_simulation(place_edge, cancel_edge)
                
        # Return best found parameters
        return {
            'best_place_edge': self.best_place_edge,
            'best_cancel_edge': self.best_cancel_edge,
            'best_pnl': self.best_pnl,
            'final_place_edge': place_edge,
            'final_cancel_edge': cancel_edge,
            'final_pnl': final_results['pnl'],
            'history': self.history,
            'converged': best_result.success,
            'iterations': best_result.nfev
        }

def extract_symbol_from_path(path: str) -> str:
    """
    Extract the stock symbol from a file path.
    """
    # Try to extract using regex pattern
    symbol_match = re.search(r'\.([A-Z]+)\.bin$', path)
    if symbol_match:
        return symbol_match.group(1)
    
    # If regex fails, just use the filename as fallback
    basename = os.path.basename(path)
    return basename.split('.')[0]

def save_results_to_file(results: Dict, symbol: str, timestamp: str = None) -> str:
    """
    Save optimization results to a single cumulative file per symbol.
    """
    # Create timestamp if not provided
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    results_dir = "/home/vir/optimization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create filename with stock symbol
    filename = f"{results_dir}/{symbol}_optimization_results.txt"
    
    # Prepare the result content
    result_content = f"\n{'=' * 50}\n"
    result_content += f"OPTIMIZATION RUN: {timestamp}\n"
    result_content += f"{'=' * 50}\n\n"
    
    result_content += f"BEST PARAMETERS:\n"
    result_content += f"Place Edge Percent: {results['best_place_edge']:.6f}%\n"
    result_content += f"Cancel Edge Percent: {results['best_cancel_edge']:.6f}%\n"
    result_content += f"P&L: ${results['best_pnl']:.2f}\n\n"
    
    if results['best_place_edge'] != results['final_place_edge']:
        result_content += f"Note: Best parameters were found during iteration, not at convergence.\n"
        result_content += f"Final parameters: place_edge={results['final_place_edge']:.6f}%, "
        result_content += f"cancel_edge={results['final_cancel_edge']:.6f}%, P&L=${results['final_pnl']:.2f}\n\n"
    
    result_content += f"Total iterations: {results['iterations']}\n"
    result_content += f"Converged: {'Yes' if results['converged'] else 'No'}\n\n"
    
    # Write results to file
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a') as f:
        # Write header if it's a new file
        if not file_exists:
            f.write(f"===== THEO STRATEGY OPTIMIZATION RESULTS - {symbol} =====\n")
            f.write(f"File created: {timestamp}\n\n")
            f.write("This file contains cumulative optimization results.\n")
            f.write("Newest results are appended at the bottom.\n\n")
            f.write(f"{'-' * 70}\n\n")
        
        # Append the new results
        f.write(result_content)
        
    print(f"Results appended to: {filename}")
    return filename

def main():
    """
    Main function to run the parameter optimization.
    """
    parser = argparse.ArgumentParser(description='Optimize TheoStrategy parameters using L-BFGS-B')
    
    parser.add_argument('--simulator', type=str, required=True,
                        help='Path to the fill simulator executable')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to the base config file')
    parser.add_argument('--book-events', type=str, default=DEFAULT_BOOK_EVENTS_PATH,
                        help='Path to the book events file')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_PATH,
                        help='Path for simulator output')
    parser.add_argument('--iterations', type=int, default=MAX_ITERATIONS,
                        help='Maximum number of iterations')
    parser.add_argument('--initial-place-edge', type=float, default=0.5,
                        help='Initial place edge percent')
    parser.add_argument('--initial-cancel-edge', type=float, default=0.2,
                        help='Initial cancel edge percent')
    
    args = parser.parse_args()
    
    # Ensure the cancel edge is less than place edge
    if args.initial_cancel_edge >= args.initial_place_edge:
        args.initial_cancel_edge = args.initial_place_edge * 0.8
        print(f"Adjusted initial cancel edge to {args.initial_cancel_edge} to ensure it's less than place edge")
    
    # Extract stock symbol from book events path
    symbol = extract_symbol_from_path(args.book_events)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print("\n==== TheoStrategy Optimizer - L-BFGS-B ====")
    print(f"Maximum iterations: {args.iterations}")
    print(f"Initial place edge: {args.initial_place_edge}%")
    print(f"Initial cancel edge: {args.initial_cancel_edge}%")
    print("=" * 43)
    
    # Create and run the optimizer
    optimizer = LBFGSOptimizer(
        simulator_path=args.simulator,
        config_path=args.config,
        book_events_path=args.book_events,
        output_path=args.output,
        initial_place_edge=args.initial_place_edge,
        initial_cancel_edge=args.initial_cancel_edge
    )
    
    results = optimizer.run_optimization(max_iterations=args.iterations)
    
    # Print results
    print("\n===== OPTIMIZATION RESULTS =====")
    print(f"Best Place Edge Percent: {results['best_place_edge']:.4f}%")
    print(f"Best Cancel Edge Percent: {results['best_cancel_edge']:.4f}%")
    print(f"Best P&L: ${results['best_pnl']:.2f}")
    if results['best_place_edge'] != results['final_place_edge']:
        print("\nNote: The best parameters were found during iteration, not at convergence.")
        print(f"Final parameters: place_edge={results['final_place_edge']:.4f}%, "
              f"cancel_edge={results['final_cancel_edge']:.4f}%, "
              f"P&L=${results['final_pnl']:.2f}")
    print(f"Total iterations: {results['iterations']}")
    print(f"Converged: {'Yes' if results['converged'] else 'No'}")
    print("===============================")

    # Save results to file
    save_results_to_file(results, symbol, timestamp)

if __name__ == "__main__":
    main()