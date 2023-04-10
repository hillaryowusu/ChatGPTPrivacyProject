output_file = "extracted_data.txt"

with open(output_file, "w") as f:
    for item in data:
        f.write(f"{item}\n")
