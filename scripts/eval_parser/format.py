import json
import csv

input_file = 'consolidated_objects.json'
json_file = 'output.json'
csv_file = 'output.csv'

# Function to filter out unwanted keys
def filter_keys(data):
    for key, obj in data.items():
        data[key] = {
            "micro ASR (lower)": obj.get("harmbench:harmbench_classifier", {}).get("micro ASR (lower)", None),
            "micro harm (lower)": obj.get("wildguardtest:harmbench_classifier", {}).get("micro harm (lower)", None),
            "overall_accuracy": obj.get("xstest", {}).get("overall_accuracy", None)
        }
    return data

def format_to_list(data):
  lst = []
  for key, obj in data.items():
    lst.append({
      "directory": key,
      "micro ASR (lower)": obj["micro ASR (lower)"],
      "micro harm (lower)": obj["micro harm (lower)"],
      "overall_accuracy": obj["overall_accuracy"]
    })
  return lst

# Main function to read, process, and write JSON
def process_json():
  # Read the input JSON file
  with open(input_file, 'r') as f:
    data = json.load(f)

  # Filter the data based on the conditions
  filtered_data = filter_keys(data)

  # Write the filtered data to the output JSON file
  with open(json_file, 'w') as f:
    json.dump(filtered_data, f, indent=2)

  list_data = format_to_list(filtered_data)
  # Write the filtered data to the output CSV file
  with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list_data[0].keys())
    writer.writeheader()  # Write header row
    writer.writerows(list_data)  # Write data rows

# Process the JSON
process_json()
