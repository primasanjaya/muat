#!/bin/bash
#
# Annotate a mutation file with the 5th column of a BED file.
#
# Note: header is expected in both inputs.
#
# dependencies: bedmap and bgzip

# Function to check and install dependencies
check_dependencies() {
    local missing_deps=()
    
    # Check for zcat/gzcat
    if ! command -v zcat >/dev/null 2>&1 && ! command -v gzcat >/dev/null 2>&1; then
        missing_deps+=("zcat/gzcat")
    fi
    
    # Check for bedmap
    if ! command -v bedmap >/dev/null 2>&1; then
        missing_deps+=("bedmap")
    fi
    
    # Check for bgzip
    if ! command -v bgzip >/dev/null 2>&1; then
        missing_deps+=("bgzip")
    fi
    
    # If there are missing dependencies, provide installation instructions
    if [ ${#missing_deps[@]} -ne 0 ]; then
        # Detect OS
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install gzip bedtools htslib
        else
            # Assume Linux (Ubuntu/Debian)
            apt-get update && apt-get install -y gzip bedtools tabix
        fi
        exit 1
    fi
}

# Run dependency check
check_dependencies

# Set ZCAT variable based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    ZCAT="gzcat"
else
    ZCAT="zcat"  # for Linux systems
fi


if [[ -z "$3" ]]; then
  echo "Usage: $0 mutations annotations output [column_name]"
  exit 2
fi

if [[ -z "$4" ]]; then
  label=$(basename ${2%.bed.gz})
else
  label=$4
fi

set -euo pipefail

muttsv=$1
annbed=$2
outtsv=$3

tab=$'\t'
IFS=" "  # preserve tabs in header
echo "Extracting header from ${muttsv} ..."
hdr=$(cat <(${ZCAT} ${muttsv}|head -n 1) )
hdr="${hdr}${tab}${label}"
echo "Annotating ${muttsv} with ${annbed} and writing to ${outtsv} ..."
date
# write header + input in TSV format with mean of annotation overlap for each mutation
cat <(echo ${hdr}) <(bedmap --sweep-all --delim '\t' --bp-ovr 1 --faster --echo --mean \
  <(gunzip -c ${muttsv}|tail -n +2|awk 'BEGIN{FS=OFS="\t"} {$2 = $2 OFS $2+1} 1') <(gunzip -c ${annbed}|tail -n +2) | sed 's/NAN/nan/' | cut -f 1-2,4-) | bgzip -c >${outtsv}

echo "done"
date