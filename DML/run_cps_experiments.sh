#!/bin/bash
#
# Run DML synthetic experiments for CPS comparison group
# This runs the three standard covariate configurations (Columns 1, 3, 8)
#

GROUP="cps"  # Change to 'cps2' or 'cps3' if desired

echo "Running DML Synthetic Experiments - CPS Group"
echo ""
echo "Group: $GROUP"
echo "This will run 3 configurations matching CVCI paper Columns 1, 3, 8"
echo ""

echo "COLUMN 1: No covariates (treatment only)"
python lalonde_synthetic_dml.py --group $GROUP
echo ""
echo "Column 1 complete!"
echo ""

echo "COLUMN 3: Demographics + RE75"
python lalonde_synthetic_dml.py --group $GROUP \
    --variables age age2 education nodegree black hispanic re75
echo ""
echo "Column 3 complete!"
echo ""

echo "COLUMN 8: All covariates"
python lalonde_synthetic_dml.py --group $GROUP \
    --variables age education nodegree black hispanic married re75 u75 u74 re74
echo ""
echo "Column 8 complete!"
echo ""

echo "ALL CPS EXPERIMENTS COMPLETE!"
echo ""
echo "Files created in ./YYYY-MM-DD/ directory:"
echo "  - lalonde_synthetic_dml_${GROUP}_[].json"
echo "  - lalonde_synthetic_dml_${GROUP}_['age', 'age2', ..., 're75'].json"
echo "  - lalonde_synthetic_dml_${GROUP}_['age', 'education', ..., 're74'].json"
echo ""
echo "Now create plots with:"
echo "  python plot_synthetic_dml_results.py ./20*/lalonde_synthetic_dml_${GROUP}_*.json --group-label CPS"
echo ""
