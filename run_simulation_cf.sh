#!/bin/bash
# Run varying_nobs CF simulation in background with caffeinate
#
# Usage:
#   ./run_simulation_cf.sh e          # Run panel e (N^exp=50) with CF
#   ./run_simulation_cf.sh f          # Run panel f (N^exp=1000) with CF
#   ./run_simulation_cf.sh e quick    # Quick mode (CF)
#   ./run_simulation_cf.sh e ultra    # Ultra-quick mode (CF)
#
# Check progress:
#   tail -f simulation_panel_cf_*.log

PANEL=${1:-e}
MODE=${2:-full}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="simulation_panel_cf_${PANEL}_${TIMESTAMP}.log"

echo "Starting CF simulation for Panel ${PANEL}"
echo "Log file: ${LOGFILE}"

# Build command based on mode
if [ "$MODE" == "ultra" ]; then
    CMD="python varying_nobs_cf.py --panel $PANEL --ultra-quick"
    echo "Mode: ULTRA-QUICK (~10-15 min, CF)"
elif [ "$MODE" == "quick" ]; then
    CMD="python varying_nobs_cf.py --panel $PANEL --quick"
    echo "Mode: QUICK (~1-2 hours, CF)"
else
    CMD="python varying_nobs_cf.py --panel $PANEL --n-sims 20 --lambda-bin 15"
    echo "Mode: FULL (~3-4 hours, CF)"
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
echo $PID > "simulation_pid_cf_${PANEL}.txt"

echo ""
echo "CF simulation running in background!"
echo ""
echo "To check progress:"
echo "  tail -f $LOGFILE"
echo ""
echo "To check saved files:"
echo "  ls simulation_results_nobs_panel_${PANEL}_cf/*.json | wc -l"
echo ""
echo "To stop:"
echo "  kill $PID"
echo ""
echo "To check if still running:"
echo "  ps aux | grep $PID"
echo ""
