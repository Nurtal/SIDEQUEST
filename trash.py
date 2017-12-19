



import glob


files_to_process = glob.glob("*.fastq") 

for data_file_name in files_to_process:

	print "[RUNNING] => split "+str(data_file_name)

	output_read_1_name = data_file_name.split(".")
	output_read_1_name = str(output_read_1_name[0])+"_r1.fastq"

	output_read_2_name = data_file_name.split(".")
	output_read_2_name = str(output_read_2_name[0])+"_r2.fastq"


	output_read_1 = open(output_read_1_name, "w")
	output_read_2 = open(output_read_2_name, "w")

	current_read = "choucroute"

	input_data = open(data_file_name, "r")
	for line in input_data:

		line_in_array = line.split(" ")
		last_element = line_in_array[-1].split("=")

		if(line[0] == "@" and last_element[0] == "length"):
			
			read_id = line_in_array[0]
			read_id_in_array = read_id.split(".")
			elt_to_check = read_id_in_array[-1]

			if(elt_to_check == "1"):
				current_read = 1
				output_read_1.write(line)
			elif(elt_to_check == "2"):
				current_read = 2
				output_read_2.write(line)
			else:
				print ">Gourgandaine<"

		if(current_read == 1):
			output_read_1.write(line)
		elif(current_read == 2):
			output_read_2.write(line)
		else:
			print "[ERROR] => can't find read "+str(current_read)

	input_data.close()

	output_read_2.close()
	output_read_1.close()

print "[EOF] => DONE"
