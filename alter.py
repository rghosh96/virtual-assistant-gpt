import csv

def append_numbers_to_csv(input_file, output_file):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Iterate through each row in the data and append a number in increasing order
    for index, row in enumerate(data):
        row[0] = f"{row[0]} {index}"  # Appending number to the end of each item

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Example usage:
input_file = 'dataset.csv'  # Replace 'input.csv' with your CSV file name
output_file = 'output.csv'  # Replace 'output.csv' with your desired output file name

append_numbers_to_csv(input_file, output_file)
print("CSV file successfully processed.")