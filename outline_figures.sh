#!/usr/bin/env bash
# Convert figure text to vector outlines so the PDFs embed no CID/Identity-H
# fonts (AAAI forbids them, even inside figures). Run after make_hero_figures.py
# and make_figures.py. Requires ghostscript.
set -e
for f in fig_hero fig_twometric fig3_coupling; do
  gs -o "paper/${f}_o.pdf" -dNoOutputFonts -sDEVICE=pdfwrite "${f}.pdf" >/dev/null
  mv "paper/${f}_o.pdf" "paper/${f}.pdf"
  echo "outlined paper/${f}.pdf"
done
