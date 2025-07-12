#!/usr/bin/env python3
"""
Parameter optimizer for TheoStrategy using Simulated Annealing.
This script finds optimal placeEdgePercent and cancelEdgePercent parameters.
"""

import subprocess
import re
import random
import math
import argparse
import os
from typing import Tuple, Dict, List

# Configuration
DEFAULT_CONFIG_PATH = "/home/vir/fill-simulator/latencies/latency_config_queue.toml"
DEFAULT_BOOK_EVENTS_PATH = "/data/20220801/nasdaq/NASDAQ.book_events.AAPL.bin"
DEFAULT_OUTPUT_PATH = "/data/20220801/nasdaq/fillsimulations/theo_strategy.AAPL.bin"
MAX_ITERATIONS = 100
INITIAL_TEMPERATURE = 100.0
COOLING_RATE = 0.98

class TheoParameterOptimizer:
    """
    Optimizer for TheoStrategy parameters using Simulated Annealing.
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
        self.current_place_edge = initial_place_edge
        self.current_cancel_edge = initial_cancel_edge
        self.best_place_edge = initial_place_edge
        self.best_cancel_edge = initial_cancel_edge
        self.best_pnl = float('-inf')
        self.history: List[Dict] = []

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
            stdout, stderr = process.communicate(input="2\n")
            
            if process.returncode != 0:
                print(f"Error running simulator: {stderr}")
                return {'pnl': float('-inf'), 'fill_rate': 0, 'orders': 0}
            
            # Parse results
            results = self.parse_simulation_results(stdout)
            
            # Clean up temp config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            
            return results
        except Exception as e:
            print(f"Exception running simulation: {e}")
            return {
                'pnl': float('-inf'),
                'fill_rate': 0,
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
    
    def generate_neighbor(self, temperature: float) -> Tuple[float, float]:
        """
        Generate neighbor with adaptive step size that decreases with temperature.
        """
        temp_factor = min(1.0, temperature / INITIAL_TEMPERATURE * 3)
        place_step = 0.2 * temp_factor
        cancel_step = 0.1 * temp_factor
        
        if random.random() < 0.5:
            new_place_edge = max(0.01, min(5.0, self.current_place_edge + random.uniform(-place_step, place_step)))
            new_cancel_edge = min(self.current_cancel_edge, new_place_edge * 0.95)
        else:
            max_cancel = self.current_place_edge * 0.95
            new_cancel_edge = max(0.005, min(max_cancel, self.current_cancel_edge + random.uniform(-cancel_step, cancel_step)))
            new_place_edge = self.current_place_edge
        
        return new_place_edge, new_cancel_edge
    
    def calculate_fitness(self, results: Dict) -> float:
        """
        Calculate fitness score for simulation results with risk adjustment.
        """
        pnl = results['pnl']
        fill_rate = results['fill_rate']
        orders_placed = results['orders_placed']
        orders_filled = results.get('orders_filled', 0)
        
        if orders_placed < 10:
            return float('-inf')
        
        # Calculate sharpe-like ratio if we have enough filled orders
        if orders_filled >= 5:
            # Estimate average P&L per trade
            avg_pnl_per_trade = pnl / orders_filled if orders_filled > 0 else 0
            
            # Use a default standard deviation since we don't have individual trade P&Ls
            default_std_dev = abs(avg_pnl_per_trade) * 0.5
            std_dev = max(0.01, default_std_dev)
            
            # Calculate risk-adjusted return component
            risk_adjusted_return = avg_pnl_per_trade / std_dev
            risk_component = risk_adjusted_return * 10
        else:
            risk_component = 0
        
        # Fill rate components
        if fill_rate < 0.1:
            fill_rate_penalty = -100
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
            # Scale based on whether it's profit or loss
            if pnl > 0:
                pnl_efficiency = pnl_per_filled * 10.0
            else:
                pnl_efficiency = pnl_per_filled * 5.0
        
        # Combine components with appropriate weights
        fitness = (pnl * 3.0) + fill_rate_penalty + fill_rate_bonus + (risk_component * 5.0) + pnl_efficiency
        
        return fitness
    
    def run_optimization(self, iterations: int = MAX_ITERATIONS) -> Dict:
        """
        Run the simulated annealing optimization process with periodic reheating.
        """
        temperature = INITIAL_TEMPERATURE
        min_temperature = 10.0  # Threshold for reheating
        
        # Evaluate initial solution
        current_results = self.run_simulation(self.current_place_edge, self.current_cancel_edge)
        current_fitness = self.calculate_fitness(current_results)
        
        # Keep track of the best solution
        if current_fitness > self.best_pnl:
            self.best_pnl = current_fitness
            self.best_place_edge = self.current_place_edge
            self.best_cancel_edge = self.current_cancel_edge
        
        # Add initial state to history
        self.history.append({
            'iteration': 0,
            'place_edge': self.current_place_edge,
            'cancel_edge': self.current_cancel_edge,
            'pnl': current_results['pnl'],
            'fill_rate': current_results['fill_rate'],
            'temperature': temperature,
            'accepted': True
        })
        
        print(f"Initial state: place_edge={self.current_place_edge:.4f}, "
            f"cancel_edge={self.current_cancel_edge:.4f}, "
            f"P&L=${current_results['pnl']:.2f}, "
            f"Fill Rate={current_results['fill_rate']:.2f}%")
        
        for i in range(iterations):
            # Check if temperature is too low
            if temperature < min_temperature:
                # Reset temperature
                temperature = INITIAL_TEMPERATURE * 0.8
                
                # Jump to random position
                random_place_edge = random.uniform(0.001, 1.0)
                random_cancel_edge = random.uniform(0.0005, random_place_edge * 0.9)
                                
                # Update current position
                self.current_place_edge = random_place_edge
                self.current_cancel_edge = random_cancel_edge
                
                # Re-evaluate at new position
                current_results = self.run_simulation(self.current_place_edge, self.current_cancel_edge)
                current_fitness = self.calculate_fitness(current_results)
                
                print(f"\nTEMPERATURE RESET at iteration {i+1}")
                print(f"New position: place_edge={self.current_place_edge:.4f}, "
                    f"cancel_edge={self.current_cancel_edge:.4f}")
                print(f"New temperature: {temperature:.4f}\n")
                
                # Add reset event to history
                self.history.append({
                    'iteration': i + 1,
                    'place_edge': self.current_place_edge,
                    'cancel_edge': self.current_cancel_edge,
                    'pnl': current_results['pnl'],
                    'fill_rate': current_results['fill_rate'],
                    'temperature': temperature,
                    'accepted': True,
                    'reset': True
                })
                
                # Skip the rest of this iteration
                continue
            
            # Generate a neighboring solution
            new_place_edge, new_cancel_edge = self.generate_neighbor(temperature)
            
            # Evaluate the new solution
            new_results = self.run_simulation(new_place_edge, new_cancel_edge)
            new_fitness = self.calculate_fitness(new_results)
            
            # Decide whether to accept the new solution
            delta_fitness = new_fitness - current_fitness

            # Safety check to prevent overflow
            if delta_fitness > 0:
                acceptance_probability = 1.0
            else:
                # Cap the exponent to avoid overflow
                exponent = max(-700.0, delta_fitness / temperature)
                acceptance_probability = math.exp(exponent)
            
            accepted = random.random() <= acceptance_probability
            
            if accepted:
                # Accept the new solution
                self.current_place_edge = new_place_edge
                self.current_cancel_edge = new_cancel_edge
                current_fitness = new_fitness
                
                # Update the best solution if needed
                if new_fitness > self.best_pnl:
                    self.best_pnl = new_fitness
                    self.best_place_edge = new_place_edge
                    self.best_cancel_edge = new_cancel_edge
            
            # Add to history
            self.history.append({
                'iteration': i + 1,
                'place_edge': new_place_edge,
                'cancel_edge': new_cancel_edge,
                'pnl': new_results['pnl'],
                'fill_rate': new_results['fill_rate'],
                'temperature': temperature,
                'accepted': accepted,
                'reset': False
            })
            
            # Adaptive cooling - slows down cooling when finding improvements
            if accepted and new_fitness > self.best_pnl * 1.05:
                temperature = min(temperature * 1.1, INITIAL_TEMPERATURE * 0.5)
            else:
                temperature *= COOLING_RATE
            
            # Print progress every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}: place_edge={new_place_edge:.4f}, "
                    f"cancel_edge={new_cancel_edge:.4f}, "
                    f"P&L=${new_results['pnl']:.2f}, "
                    f"Fill Rate={new_results['fill_rate']:.2f}%, "
                    f"Accepted: {accepted}, "
                    f"Temperature: {temperature:.4f}")

            # Print best results so far after every 50 iterations
            if (i + 1) % 50 == 0:
                print(f"\n----- BEST RESULTS SO FAR (ITERATION {i+1}) -----")
                print(f"Best Place Edge: {self.best_place_edge:.4f}%")
                print(f"Best Cancel Edge: {self.best_cancel_edge:.4f}%")
                print(f"Best P&L: ${self.best_pnl:.2f}")
                print(f"--------------------------------------------\n")
        
        # Return the best found parameters
        return {
            'best_place_edge': self.best_place_edge,
            'best_cancel_edge': self.best_cancel_edge,
            'best_pnl': self.best_pnl,
            'history': self.history
        }
    
def main():
    """
    Main function to run the parameter optimization.
    """
    parser = argparse.ArgumentParser(description='Optimize TheoStrategy parameters using Simulated Annealing')
    
    parser.add_argument('--simulator', type=str, required=True,
                        help='Path to the fill simulator executable')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to the base config file')
    parser.add_argument('--book-events', type=str, default=DEFAULT_BOOK_EVENTS_PATH,
                        help='Path to the book events file')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_PATH,
                        help='Path for simulator output')
    parser.add_argument('--iterations', type=int, default=MAX_ITERATIONS,
                        help='Number of iterations to run')
    parser.add_argument('--initial-place-edge', type=float, default=0.5,
                        help='Initial place edge percent')
    parser.add_argument('--initial-cancel-edge', type=float, default=0.2,
                        help='Initial cancel edge percent')
    
    args = parser.parse_args()
    
    # Ensure the cancel edge is less than place edge
    if args.initial_cancel_edge >= args.initial_place_edge:
        args.initial_cancel_edge = args.initial_place_edge * 0.8
        print(f"Adjusted initial cancel edge to {args.initial_cancel_edge} to ensure it's less than place edge")
    
    # Create and run the optimizer
    optimizer = TheoParameterOptimizer(
        simulator_path=args.simulator,
        config_path=args.config,
        book_events_path=args.book_events,
        output_path=args.output,
        initial_place_edge=args.initial_place_edge,
        initial_cancel_edge=args.initial_cancel_edge
    )
    
    print(f"Starting optimization with {args.iterations} iterations...")
    results = optimizer.run_optimization(iterations=args.iterations)
    
    # Print and save the results
    print("\n===== OPTIMIZATION RESULTS =====")
    print(f"Best Place Edge Percent: {results['best_place_edge']:.4f}%")
    print(f"Best Cancel Edge Percent: {results['best_cancel_edge']:.4f}%")
    print(f"Best P&L: ${results['best_pnl']:.2f}")

if __name__ == "__main__":
    main()