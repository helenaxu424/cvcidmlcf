#!/bin/bash
# Run varying_nobs simulation in background with caffeinate
#
# Usage:
#   ./run_simulation.sh e          # Run panel e (N^exp=50)
#   ./run_simulation.sh f          # Run panel f (N^exp=1000)
#   ./run_simulation.sh e quick    # Quick mode
#   ./run_simulation.sh e ultra    # Ultra-quick mode
#
# Check progress:
#   tail -f simulation_panel_*.log
#   ./check_progress.sh

PANEL=${1:-e}
MODE=${2:-full}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="simulation_panel_${PANEL}_${TIMESTAMP}.log"

echo "=================================================="
echo "Starting simulation for Panel ${PANEL}"
echo "Log file: ${LOGFILE}"
echo "=================================================="

# Build command based on mode
if [ "$MODE" == "ultra" ]; then
    CMD="python varying_nobs_dml.py --panel $PANEL --ultra-quick"
    echo "Mode: ULTRA-QUICK (~10-15 min)"
elif [ "$MODE" == "quick" ]; then
    CMD="python varying_nobs_dml.py --panel $PANEL --quick"
    echo "Mode: QUICK (~1-2 hours)"
else
    CMD="python varying_nobs_dml.py --panel $PANEL --n-sims 20 --lambda-bin 15"
    echo "Mode: FULL (~3-4 hours)"
fi

echo "Command: $CMD"
echo ""

# Run with caffeinate in background
echo "Starting at $(date)" > "$LOGFILE"
echo "Command: $CMD" >> "$LOGFILE"
echo "==========================================" >> "$LOGFILE"

caffeinate -i $CMD >> "$LOGFILE" 2>&1 &
PID=$!

echo "Process started with PID: $PID"
echo $PID > "simulation_pid_${PANEL}.txt"

echo ""
echo "=================================================="
echo "Simulation running in background!"
echo "=================================================="
echo ""
echo "To check progress:"
echo "  tail -f $LOGFILE"
echo ""
echo "To check saved files:"
echo "  ls simulation_results_nobs_panel_${PANEL}/*.json | wc -l"
echo ""
echo "To stop:"
echo "  kill $PID"
echo ""
echo "To check if still running:"
echo "  ps aux | grep $PID"
echo ""
