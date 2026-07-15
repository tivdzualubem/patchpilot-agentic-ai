#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
latexmk -pdf -interaction=nonstopmode -halt-on-error PatchPilot_Final_Report.tex
pdfinfo PatchPilot_Final_Report.pdf | sed -n '1,20p'
echo "REPORT_BUILD_COMPLETE=1"
