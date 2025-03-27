#!/bin/bash

parent_dir=$(pwd)

# Function to consolidate JSON files of the same name from all directories
consolidate_jsons() {
  local filename=$1
  local parent_dir=$2
  local consolidated_json="{"

  # Find all directories containing the file
  for dir in "$parent_dir"/*/; do
    if [ -d "$dir" ]; then
      json_file="$dir/$filename"
      
      # Check if the JSON file exists in the directory
      if [ -f "$json_file" ]; then
        # Get the directory name (without path)
        dir_name=$(basename "$dir")
        
        # Read the content of the JSON file
        json_content=$(<"$json_file")
        
        # Add the directory name and JSON content to the consolidated JSON object
        if [ "$consolidated_json" != "{" ]; then
          consolidated_json="$consolidated_json, "
        fi
        consolidated_json="$consolidated_json\"$dir_name\": $json_content"
      fi
    fi
  done

  # Close the JSON object
  consolidated_json="$consolidated_json}"

  # Output consolidated JSON to the parent folder
  echo "$consolidated_json" > "$parent_dir/$filename"
  echo "Consolidated $filename saved to $parent_dir/$filename"
}

# Loop through each JSON file in the parent directory
for json_file in "$parent_dir"/*/*.json; do
  if [ -f "$json_file" ]; then
    filename=$(basename "$json_file")
    consolidate_jsons "$filename" "$parent_dir"
  fi
done

# Consolidates the 3 hardcoded json files into one
# TODO: Make file names dynamic?
files=("harmbench:harmbench_classifier.json" "wildguardtest:harmbench_classifier.json" "xstest.json")
jq -s 'reduce .[] as $item ({}; 
  # For each object in the array, merge its contents
  . * $item)' "${files[@]}" > consolidated_objects.json

python3 format.py

rm consolidated_objects.json

echo "script completed"
