#!/usr/bin/env python3
"""
This script finds optimal parameters for all book events files in a directory using L-BFGS-B algorithm.
"""

import os
import argparse
import glob
import time
import subprocess
import concurrent.futures
from datetime import datetime
import re

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

def run_optimization(args):
    """
    Run L-BFGS-B optimization for a single book events file.
    """
    simulator_path, config_path, book_events_path, output_dir, iterations, place_edge, cancel_edge = args
    
    # Extract symbol from book events path
    symbol = extract_symbol_from_path(book_events_path)
    
    # Create output path for this symbol
    output_path = f"{output_dir}/theo_strategy.{symbol}.bin"
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting optimization for {symbol}")
    
    # Construct command to run the optimizer
    cmd = [
        "python", "L-BFGS-B_algorithm.py",
        "--simulator", simulator_path,
        "--config", config_path,
        "--book-events", book_events_path,
        "--output", output_path,
        "--iterations", str(iterations),
        "--initial-place-edge", str(place_edge),
        "--initial-cancel-edge", str(cancel_edge)
    ]
    
    # Run the optimization process
    try:
        subprocess.run(cmd, check=True)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed optimization for {symbol}")
        return (True, symbol)
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error optimizing {symbol}: {e}")
        return (False, symbol)
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unexpected error for {symbol}: {e}")
        return (False, symbol)

def main():
    """
    Main function to run batch optimization.
    """
    parser = argparse.ArgumentParser(description='Batch optimize TheoStrategy parameters for multiple symbols')
    
    parser.add_argument('--book-events-dir', type=str, required=True,
                        help='Directory containing book events files')
    parser.add_argument('--simulator', type=str, required=True,
                        help='Path to the fill simulator executable')
    parser.add_argument('--config', type=str, default="/home/vir/fill-simulator/latencies/latency_config_queue.toml",
                        help='Path to the base config file')
    parser.add_argument('--output-dir', type=str, default="/data/20220801/nasdaq/fillsimulations",
                        help='Directory for simulator outputs')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Maximum iterations per optimization')
    parser.add_argument('--initial-place-edge', type=float, default=0.5,
                        help='Initial place edge percent')
    parser.add_argument('--initial-cancel-edge', type=float, default=0.2,
                        help='Initial cancel edge percent')
    parser.add_argument('--pattern', type=str, default="*.book_events.*.bin",
                        help='Glob pattern to match book events files')
    parser.add_argument('--max-workers', type=int, default=1,
                        help='Maximum number of parallel optimizations')
    
    args = parser.parse_args()
    
    # Make sure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all book events files
    book_events_pattern = os.path.join(args.book_events_dir, args.pattern)
    book_events_files = sorted(glob.glob(book_events_pattern))
    
    if not book_events_files:
        print(f"No book events files found matching pattern '{args.pattern}' in {args.book_events_dir}")
        return
    
    print(f"Found {len(book_events_files)} book events files to process:")
    for i, file_path in enumerate(book_events_files):
        symbol = extract_symbol_from_path(file_path)
        print(f"  {i+1}. {symbol}: {os.path.basename(file_path)}")
    
    start_time = time.time()
    
    # Prepare arguments for each optimization task
    tasks = [
        (args.simulator, args.config, book_events_path, args.output_dir, 
         args.iterations, args.initial_place_edge, args.initial_cancel_edge)
        for book_events_path in book_events_files
    ]
    
    # Run optimizations in parallel or sequentially
    results = []
    if args.max_workers > 1:
        print(f"\nRunning optimizations in parallel with {args.max_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            results = list(executor.map(run_optimization, tasks))
    else:
        print("\nRunning optimizations sequentially...")
        for task in tasks:
            results.append(run_optimization(task))
    
    # Calculate statistics
    successful = [symbol for success, symbol in results if success]
    failed = [symbol for success, symbol in results if not success]
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print summary
    print("\n" + "=" * 50)
    print("BATCH OPTIMIZATION SUMMARY")
    print("=" * 50)
    print(f"Total book events files processed: {len(book_events_files)}")
    print(f"Successfully optimized: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("=" * 50)
    
    if failed:
        print("\nFailed optimizations:")
        for symbol in failed:
            print(f"  - {symbol}")

if __name__ == "__main__":
    main()