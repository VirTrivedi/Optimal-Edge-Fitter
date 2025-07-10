#!/usr/bin/env python3
"""
Parameter optimizer for TheoStrategy using an Evolutionary Algorithm.
This script finds optimal placeEdgePercent and cancelEdgePercent parameters.
"""

import subprocess
import re
import random
import argparse
import os
import statistics
import time
from typing import Tuple, Dict, List

# Configuration
DEFAULT_CONFIG_PATH = "/home/vir/fill-simulator/latencies/latency_config_queue.toml"
DEFAULT_BOOK_EVENTS_PATH = "/data/20220801/nasdaq/NASDAQ.book_events.AAPL.bin"
DEFAULT_OUTPUT_PATH = "/data/20220801/nasdaq/fillsimulations/theo_strategy.AAPL.bin"
DEFAULT_POPULATION_SIZE = 20
DEFAULT_GENERATIONS = 10
DEFAULT_MUTATION_RATE = 0.2
DEFAULT_CROSSOVER_RATE = 0.7


class Individual:
    """Represents a candidate solution in the population."""
    
    def __init__(self, place_edge: float, cancel_edge: float):
        """Initialize an individual with place_edge and cancel_edge parameters."""
        self.place_edge = place_edge
        self.cancel_edge = min(cancel_edge, place_edge * 0.95)
        self.fitness = float('-inf')
        self.pnl = 0.0
        self.fill_rate = 0.0
        
    def __str__(self):
        return f"place_edge={self.place_edge:.4f}, cancel_edge={self.cancel_edge:.4f}, fitness={self.fitness:.2f}"


class EvolutionaryOptimizer:
    """
    Optimizer for TheoStrategy parameters using an Evolutionary Algorithm.
    """
    def __init__(
        self, 
        simulator_path: str, 
        config_path: str, 
        book_events_path: str,
        output_path: str,
        population_size: int = DEFAULT_POPULATION_SIZE,
        mutation_rate: float = DEFAULT_MUTATION_RATE,
        crossover_rate: float = DEFAULT_CROSSOVER_RATE
    ):
        self.simulator_path = simulator_path
        self.config_path = config_path
        self.book_events_path = book_events_path
        self.output_path = output_path
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.strategy_md_latency_ns = 1000
        self.exchange_latency_ns = 10000
        self.extract_base_config_values()
        self.best_individual = None
        self.history = []
        self.evaluation_cache = {}
    
    def extract_base_config_values(self):
        """Extract static values from base config."""
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
    
    def create_initial_population(self) -> List[Individual]:
        """Create the initial population of individuals."""
        population = []
        
        # Create random individuals
        for _ in range(self.population_size):
            place_edge = random.uniform(0.01, 2.0)
            cancel_edge = random.uniform(0.005, place_edge * 0.95)
            individual = Individual(place_edge, cancel_edge)
            population.append(individual)
        
        # Add some "educated guesses" based on prior knowledge
        educated_guesses = [
            (0.1, 0.05),
            (0.5, 0.2),
            (1.0, 0.5),
            (0.25, 0.15),
            (0.05, 0.01)
        ]
        
        # Replace some random individuals with educated guesses
        for i, (place, cancel) in enumerate(educated_guesses):
            if i < len(population):
                population[i] = Individual(place, cancel)
        
        return population
    
    def evaluate_individual(self, individual: Individual) -> Dict:
        """Evaluate an individual by running a simulation."""
        # Check if we've already evaluated this individual
        cache_key = (round(individual.place_edge, 4), round(individual.cancel_edge, 4))
        if cache_key in self.evaluation_cache:
            results = self.evaluation_cache[cache_key]
            individual.fitness = self.calculate_fitness(results)
            individual.pnl = results['pnl']
            individual.fill_rate = results['fill_rate']
            return results
        
        # Create temporary config file with the parameters
        temp_config_path = self.create_temp_config(individual.place_edge, individual.cancel_edge)
        
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
            
            # Update individual's fitness
            individual.fitness = self.calculate_fitness(results)
            individual.pnl = results['pnl']
            individual.fill_rate = results['fill_rate']
            
            return results
            
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"Simulation timed out for {individual}")
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            results = {
                'pnl': float('-inf'),
                'fill_rate': 0.0,
                'orders_placed': 0,
                'orders_filled': 0
            }
            individual.fitness = float('-inf')
            return results
            
        except Exception as e:
            print(f"Exception running simulation: {e}")
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            results = {
                'pnl': float('-inf'),
                'fill_rate': 0.0,
                'orders_placed': 0,
                'orders_filled': 0
            }
            individual.fitness = float('-inf')
            return results
    
    def create_temp_config(self, place_edge: float, cancel_edge: float) -> str:
        """Create a temporary config file with the parameters."""
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
        """Parse simulation results from the output."""
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

    def select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """Select two parents using tournament selection."""
        def tournament_select(k=3):
            # Select k random individuals and take the best
            candidates = random.sample(population, k)
            return max(candidates, key=lambda ind: ind.fitness)
        
        parent1 = tournament_select()
        parent2 = tournament_select()
        
        return parent1, parent2
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover to create two offspring."""
        if random.random() < self.crossover_rate:
            alpha = random.random()
            
            # Crossover place_edge
            child1_place = alpha * parent1.place_edge + (1 - alpha) * parent2.place_edge
            child2_place = alpha * parent2.place_edge + (1 - alpha) * parent1.place_edge
            
            # Crossover cancel_edge
            child1_cancel = alpha * parent1.cancel_edge + (1 - alpha) * parent2.cancel_edge
            child2_cancel = alpha * parent2.cancel_edge + (1 - alpha) * parent1.cancel_edge
            
            # Create children
            child1 = Individual(child1_place, child1_cancel)
            child2 = Individual(child2_place, child2_cancel)
            
            return child1, child2
        else:
            # Return clones of parents if no crossover
            return Individual(parent1.place_edge, parent1.cancel_edge), Individual(parent2.place_edge, parent2.cancel_edge)
    
    def mutate(self, individual: Individual) -> None:
        """Apply mutation to an individual."""
        if random.random() < self.mutation_rate:
            # Decide which parameter to mutate, or both
            mutation_type = random.randint(0, 2)
            
            # Mutate place_edge
            if mutation_type in (0, 2):
                # Gaussian mutation
                sigma = individual.place_edge * 0.2
                new_place = individual.place_edge + random.gauss(0, sigma)
                # Ensure valid range
                new_place = max(0.01, min(5.0, new_place))
                individual.place_edge = new_place
            
            # Mutate cancel_edge
            if mutation_type in (1, 2):
                # Gaussian mutation
                sigma = individual.cancel_edge * 0.2
                new_cancel = individual.cancel_edge + random.gauss(0, sigma)
                # Ensure valid range and constraint
                new_cancel = max(0.005, min(individual.place_edge * 0.95, new_cancel))
                individual.cancel_edge = new_cancel
    
    def run_optimization(self, generations: int = DEFAULT_GENERATIONS) -> Dict:
        """Run the evolutionary algorithm optimization process."""
        start_time = time.time()
        
        # Create initial population
        population = self.create_initial_population()
        
        # Evaluate initial population
        print("Evaluating initial population...")
        for individual in population:
            self.evaluate_individual(individual)
        
        # Find the best individual in the initial population
        best_individual = max(population, key=lambda ind: ind.fitness)
        self.best_individual = best_individual
        
        # Add initial stats to history
        self.history.append({
            'generation': 0,
            'best_fitness': best_individual.fitness,
            'best_place_edge': best_individual.place_edge,
            'best_cancel_edge': best_individual.cancel_edge,
            'best_pnl': best_individual.pnl,
            'best_fill_rate': best_individual.fill_rate,
            'avg_fitness': statistics.mean([ind.fitness for ind in population if ind.fitness > float('-inf')]),
            'elapsed_time': time.time() - start_time
        })
        
        print(f"Initial best: {best_individual}")
        
        # Main evolutionary loop
        for generation in range(1, generations + 1):
            generation_start = time.time()
            print(f"Generation {generation}/{generations}...")
            
            # Create new population
            new_population = []
            
            # Keep the best individual
            new_population.append(Individual(best_individual.place_edge, best_individual.cancel_edge))
            
            # Create rest of population through selection, crossover, mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population)
                child1, child2 = self.crossover(parent1, parent2)
                
                # Apply mutation
                self.mutate(child1)
                self.mutate(child2)
                
                # Add children to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Replace population
            population = new_population
            
            # Evaluate new population
            for individual in population:
                self.evaluate_individual(individual)
            
            # Find the best individual in this generation
            generation_best = max(population, key=lambda ind: ind.fitness)
            
            # Update global best if needed
            if generation_best.fitness > best_individual.fitness:
                best_individual = generation_best
                self.best_individual = best_individual
                print(f"New best found: {best_individual}")
            
            # Add generation stats to history
            self.history.append({
                'generation': generation,
                'best_fitness': best_individual.fitness,
                'best_place_edge': best_individual.place_edge,
                'best_cancel_edge': best_individual.cancel_edge,
                'best_pnl': best_individual.pnl,
                'best_fill_rate': best_individual.fill_rate,
                'avg_fitness': statistics.mean([ind.fitness for ind in population if ind.fitness > float('-inf')]),
                'elapsed_time': time.time() - start_time
            })
            
            # Print generation stats
            gen_time = time.time() - generation_start
            print(f"Generation {generation} completed in {gen_time:.1f}s")
            print(f"Best: place_edge={best_individual.place_edge:.4f}, "
                  f"cancel_edge={best_individual.cancel_edge:.4f}, "
                  f"PnL=${best_individual.pnl:.2f}, "
                  f"Fill Rate={best_individual.fill_rate:.2f}%")
            
            # Print population stats
            valid_fitnesses = [ind.fitness for ind in population if ind.fitness > float('-inf')]
            if valid_fitnesses:
                avg_fitness = statistics.mean(valid_fitnesses)
                print(f"Population average fitness: {avg_fitness:.2f}")
                
                place_edges = [ind.place_edge for ind in population]
                cancel_edges = [ind.cancel_edge for ind in population]
                print(f"Diversity - place_edge: {statistics.stdev(place_edges):.4f}, "
                      f"cancel_edge: {statistics.stdev(cancel_edges):.4f}")
        
        # Return the results
        return {
            'best_place_edge': best_individual.place_edge,
            'best_cancel_edge': best_individual.cancel_edge,
            'best_pnl': best_individual.pnl,
            'best_fill_rate': best_individual.fill_rate,
            'history': self.history
        }


def main():
    """Main function to run the parameter optimization."""
    parser = argparse.ArgumentParser(description='Optimize TheoStrategy parameters using an Evolutionary Algorithm')
    
    parser.add_argument('--simulator', type=str, required=True,
                        help='Path to the fill simulator executable')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to the base config file')
    parser.add_argument('--book-events', type=str, default=DEFAULT_BOOK_EVENTS_PATH,
                        help='Path to the book events file')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_PATH,
                        help='Path for simulator output')
    parser.add_argument('--population', type=int, default=DEFAULT_POPULATION_SIZE,
                        help='Population size')
    parser.add_argument('--generations', type=int, default=DEFAULT_GENERATIONS,
                        help='Number of generations to run')
    parser.add_argument('--mutation-rate', type=float, default=DEFAULT_MUTATION_RATE,
                        help='Mutation rate (0.0-1.0)')
    parser.add_argument('--crossover-rate', type=float, default=DEFAULT_CROSSOVER_RATE,
                        help='Crossover rate (0.0-1.0)')
    
    args = parser.parse_args()
    
    print("\n==== TheoStrategy Optimizer - Evolutionary Algorithm ====")
    print(f"Population size: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Mutation rate: {args.mutation_rate}")
    print(f"Crossover rate: {args.crossover_rate}")
    print("=" * 56)
    
    # Create and run the optimizer
    optimizer = EvolutionaryOptimizer(
        simulator_path=args.simulator,
        config_path=args.config,
        book_events_path=args.book_events,
        output_path=args.output,
        population_size=args.population,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate
    )
    
    print(f"\nStarting optimization with {args.generations} generations...")
    results = optimizer.run_optimization(generations=args.generations)
    
    # Print results
    print("\n===== OPTIMIZATION RESULTS =====")
    print(f"Best Place Edge: {results['best_place_edge']:.4f}%")
    print(f"Best Cancel Edge: {results['best_cancel_edge']:.4f}%")
    print(f"Best P&L: ${results['best_pnl']:.2f}")
    print(f"Fill Rate: {results['best_fill_rate']:.2f}%")
    print("===============================")


if __name__ == "__main__":
    main()