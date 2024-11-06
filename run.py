# Define the file path
file_path = "/home/thinh/.local/lib/python3.10/site-packages/basicsr/data/degradations.py"

# Define the new import statement
new_import_statement = "from torchvision.transforms.functional import rgb_to_grayscale\n"

# Read the content of the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Modify the desired line (line 8 in this case)
if len(lines) >= 8:
    lines[7] = new_import_statement  # Index 7 corresponds to line 8 (0-based indexing)

# Write the modified content back to the file
with open(file_path, 'w') as file:
    file.writelines(lines)

print("Replacement completed successfully!")