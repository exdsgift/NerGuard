#!/bin/bash
# Convert all Mermaid diagrams to SVG using kroki.io API (no Chrome needed)

OUTPUT_DIR="$(dirname "$0")/svg_output"

mkdir -p "$OUTPUT_DIR/paper"
mkdir -p "$OUTPUT_DIR/process_diagrams"
mkdir -p "$OUTPUT_DIR/results"

count=0
total=$(find "$(dirname "$0")" -name "*.mmd" | wc -l)

find "$(dirname "$0")" -name "*.mmd" | while read -r file; do
    count=$((count + 1))
    filename=$(basename "$file" .mmd)
    subdir=$(basename "$(dirname "$file")")
    output="$OUTPUT_DIR/$subdir/${filename}.svg"

    echo "[$count/$total] Converting: $file"

    # POST file content to kroki.io and save SVG
    if curl -s -X POST -H "Content-Type: text/plain" --data-binary @"$file" "https://kroki.io/mermaid/svg" -o "$output"; then
        if [ -s "$output" ] && head -1 "$output" | grep -q "^<svg"; then
            echo "  -> $output"
        else
            echo "  ERROR: Invalid SVG output"
            cat "$output"
            rm -f "$output"
        fi
    else
        echo "  ERROR: curl failed"
    fi

    sleep 0.3  # Rate limiting
done

echo ""
echo "Done! SVGs saved in $OUTPUT_DIR"
echo ""
echo "File counts:"
ls "$OUTPUT_DIR"/paper/*.svg 2>/dev/null | wc -l | xargs echo "  paper:"
ls "$OUTPUT_DIR"/process_diagrams/*.svg 2>/dev/null | wc -l | xargs echo "  process_diagrams:"
ls "$OUTPUT_DIR"/results/*.svg 2>/dev/null | wc -l | xargs echo "  results:"
