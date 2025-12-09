#!/bin/bash
#
# Run Causal Forest synthetic experiments for CPS comparison group

GROUP="cps"   # Can change to cps2 or cps3

echo "Running Causal Forest (CF) Synthetic Experiments - CPS Group"
echo ""
echo "Group: $GROUP"
echo "Running 3 configurations (Columns 1, 3, 8)"
echo ""

# Column 1
echo "COLUMN 1: No covariates"
python lalonde_synthetic_cf.py --group $GROUP
echo ""

# Column 3
echo "COLUMN 3: Demographics + RE75"
python lalonde_synthetic_cf.py --group $GROUP \
    --variables age age2 education nodegree black hispanic re75
echo ""

# Column 8
echo "COLUMN 8: All covariates"
python lalonde_synthetic_cf.py --group $GROUP \
    --variables age education nodegree black hispanic married re75 u75 u74 re74
echo ""

echo "ALL CPS CF EXPERIMENTS COMPLETE!"
echo ""
echo "Output files created in ./YYYY-MM-DD/"
echo ""
echo "Now create CF plots with:"
echo "  python plot_synthetic_cf_results.py ./20*/lalonde_synthetic_cf_${GROUP}_*.json --group-label CPS"
echo ""
