#!/usr/bin/env python3
"""
This script finds optimal correlation strategy parameters for all book events files in a directory using L-BFGS-B algorithm.
"""

import os
import argparse
import glob
import time
import subprocess
import concurrent.futures
from datetime import datetime
import re
import threading
import sys
import select
import termios
import tty

# Default configurations
DEFAULT_CONFIG_PATH = "/home/vir/fill-simulator/latencies/latency_config_queue.toml"
DEFAULT_OUTPUT_DIRECTORY = "/data/20220801/nasdaq/fillsimulations"
DEFAULT_CORRELATION_PATH = "/data/20220801/nasdaq/bars/overall_correlations.csv"
DEFAULT_SYMBOL_MAPPING_PATH = "/data/20220801/nasdaq/nasdaq_20220801_symbol_map.csv"
MAX_ITERATIONS = 15  # Maximum number of function evaluations

stop_requested = False

def is_data():
    """
    Check if there is data available to be read from stdin.
    """
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def keyboard_listener():
    """
    Listen for keyboard input in a separate thread.
    """
    global stop_requested
    
    # Store the terminal settings
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        # Set the terminal to raw mode
        tty.setraw(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        
        print("Press 'q' at any time to stop the optimization process.")
        
        while True:
            if is_data():
                c = sys.stdin.read(1)
                if c == 'q':
                    print("\nStop requested. Waiting for current optimizations to complete...")
                    stop_requested = True
                    break
            time.sleep(0.1)
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def extract_symbol_from_path(path: str) -> str:
    """
    Extract the stock symbol from a file path.
    """
    # Try to extract using regex pattern
    symbol_match = re.search(r'\.book_events\.([A-Z0-9\.\+\-\=]+)\.bin$', path)
    if symbol_match:
        return symbol_match.group(1)
    
    # If regex fails, just use the filename as fallback
    basename = os.path.basename(path)
    return basename.split('.')[0]

def run_optimization(args):
    """
    Run L-BFGS-B optimization for a single book events file.
    """
    global stop_requested
    
    if stop_requested:
        return (False, "Skipped due to user request")

    simulator_path, config_path, correlation_path, symbol_mapping_path, book_events_path, output_dir, iterations, place_edge, cancel_edge, self_weight = args
    
    # Extract symbol from book events path
    symbol = extract_symbol_from_path(book_events_path)
    
    # Create output path for this symbol
    output_path = f"{output_dir}/correlation_strategy.{symbol}.bin"
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting optimization for {symbol}")
    
    # Construct command to run the optimizer
    cmd = [
        "python", "correlation_LBFGS_algorithm.py",
        "--simulator", simulator_path,
        "--config", config_path,
        "--book-events", book_events_path,
        "--correlation", correlation_path,
        "--symbol-mapping", symbol_mapping_path,
        "--output", output_path,
        "--iterations", str(iterations),
        "--initial-place-edge", str(place_edge),
        "--initial-cancel-edge", str(cancel_edge),
        "--initial-self-weight", str(self_weight)
    ]
    
    # Run the optimization process
    try:
        process = subprocess.run(cmd, check=False, timeout=600)

        if process.returncode == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed optimization for {symbol}")
            return (True, symbol)
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error optimizing {symbol} (return code {process.returncode})")
            return (False, symbol)
        
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Optimization for {symbol} timed out after 5 minutes")
        return (False, symbol)
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unexpected error for {symbol}: {e}")
        return (False, symbol)
    
def main():
    """
    Main function to run batch optimization.
    """
    parser = argparse.ArgumentParser(description='Batch optimize CorrelationStrategy parameters for multiple symbols')
    
    parser.add_argument('--book-events-dir', type=str, required=True,
                        help='Directory containing book events files')
    parser.add_argument('--simulator', type=str, required=True,
                        help='Path to the fill simulator executable')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to the base config file')
    parser.add_argument('--correlation', type=str, default=DEFAULT_CORRELATION_PATH,
                        help='Path to the correlation CSV file')
    parser.add_argument('--symbol-mapping', type=str, default=DEFAULT_SYMBOL_MAPPING_PATH,
                        help='Path to the symbol mapping CSV file')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIRECTORY,
                        help='Directory for simulator outputs')
    parser.add_argument('--iterations', type=int, default=MAX_ITERATIONS,
                        help='Maximum iterations per optimization')
    parser.add_argument('--initial-place-edge', type=float, default=0.05,
                        help='Initial place edge percent')
    parser.add_argument('--initial-cancel-edge', type=float, default=0.02,
                        help='Initial cancel edge percent')
    parser.add_argument('--initial-self-weight', type=float, default=0.5,
                        help='Initial self weight (0-1)')
    parser.add_argument('--pattern', type=str, default="*.book_events.*.bin",
                        help='Glob pattern to match book events files')
    parser.add_argument('--max-workers', type=int, default=1,
                        help='Maximum number of parallel optimizations')
    parser.add_argument('--symbols', type=str, nargs='*',
                        help='List of specific symbols to optimize (optional)')
    
    args = parser.parse_args()
    
    # Make sure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all book events files
    book_events_pattern = os.path.join(args.book_events_dir, args.pattern)
    book_events_files = sorted(glob.glob(book_events_pattern))
    
    if not book_events_files:
        print(f"No book events files found matching pattern '{args.pattern}' in {args.book_events_dir}")
        return
    
    # Filter by specific symbols if provided
    if args.symbols and len(args.symbols) > 0:
        filtered_files = []
        for file_path in book_events_files:
            symbol = extract_symbol_from_path(file_path)
            if symbol in args.symbols:
                filtered_files.append(file_path)
        
        if not filtered_files:
            print(f"None of the specified symbols {args.symbols} were found in the book events files.")
            return
        
        book_events_files = filtered_files
    
    print(f"Found {len(book_events_files)} book events files to process:")
    for i, file_path in enumerate(book_events_files):
        symbol = extract_symbol_from_path(file_path)
        print(f"  {i+1}. {symbol}: {os.path.basename(file_path)}")
    
    # Start keyboard listener thread
    listener_thread = threading.Thread(target=keyboard_listener)
    listener_thread.daemon = True
    listener_thread.start()
    
    start_time = time.time()
    
    # Prepare arguments for each optimization task
    tasks = [
        (args.simulator, args.config, args.correlation, args.symbol_mapping, book_events_path, 
         args.output_dir, args.iterations, args.initial_place_edge, 
         args.initial_cancel_edge, args.initial_self_weight)
        for book_events_path in book_events_files
    ]

    # Run optimizations in parallel or sequentially
    results = []
    if args.max_workers > 1:
        print(f"\nRunning optimizations in parallel with {args.max_workers} workers...")
        print("Note: Press 'q' at any time to stop after current jobs complete.")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for task in tasks:
                if stop_requested:
                    break
                futures.append(executor.submit(run_optimization, task))
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                if stop_requested:
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break
    else:
        print("\nRunning optimizations sequentially...")
        print("Note: Press 'q' at any time to stop after the current optimization completes.")
        
        for task in tasks:
            if stop_requested:
                print("Stopping due to user request.")
                break
            results.append(run_optimization(task))
    
    # Calculate statistics
    successful = [symbol for success, symbol in results if success]
    failed = [symbol for success, symbol in results if not success]
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print summary
    print("\n" + "=" * 50)
    print("BATCH CORRELATION STRATEGY OPTIMIZATION SUMMARY")
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
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        sys.exit(0)