

## patch fastq spliiting problem

input_file_name = "Bibi.txt"

input_data = open(input_file_name, "r")
output_data_name = input_file_name.split(".")
output_data_name = output_data_name[0]+"_reformat.fastq"

output_file = open(output_data_name, "w")
last_header_line = "tartiflete"
for line in input_data:

	if(line[0] == "@" and last_header_line != line):
		last_header_line = line
		output_file.write(line)
	elif(last_header_line != line):
		output_file.write(line)

input_data.close()
output_file.close()



