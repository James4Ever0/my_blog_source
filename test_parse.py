from parse import parse
input_date_string = "2022-11-28-22-01-29"

# Define the format to parse the input string
input_format = "{year:d}-{month:d}-{day:d}-{hour:d}-{minute:d}-{second:d}"

# Parse the input string using the specified format
parsed = parse(input_format, input_date_string)
print(type(parsed))
