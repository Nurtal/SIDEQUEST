"""
Patrice Script
"""
import glob
import numpy

# get files
data_files = glob.glob("*.csv")
result_file = open("patrice_results.csv", "w")
header = "file_name,max_MFI,min_MFI (mean first time serie),max_derivative,min_derivative,mean_pos_derivative,mean_neg_derivative,amplitude\n"
result_file.write(header)

# open file
for data in data_files:

	print "[+] Process "+str(data)

	input_data = open(data, "r")

	x_vector = []
	y_vector = []
	first_range_vector = []
	derivative_array = []

	negative_derivative = []
	postive_derivative = []

	cmpt = 0
	for line in input_data:
		line = line.split("\n")
		line = line[0]
		line_in_array = line.split(",")

		if(cmpt > 0):
			x = line_in_array[1]
			y = line_in_array[2]

			if(float(x) < 30.0):
				first_range_vector.append(float(y))
			else:
				x_vector.append(float(x))
				y_vector.append(float(y))

		cmpt += 1

	# compute measures
	max_y = max(y_vector)
	mean_first_range = numpy.mean(first_range_vector)
	derivative = numpy.gradient(y_vector)
	
	cmpt = 0
	for x in x_vector:
		f_x = y_vector[cmpt]

		if(cmpt+1 < len(x_vector)):
			h = float(x_vector[cmpt+1]) - float(x)
			f_x_h = float(y_vector[cmpt+1])
			d = (float(f_x_h)-float(f_x))/float(h)

			derivative_array.append(d)

			if(float(d) >= 0):
				postive_derivative.append(d)
			else:
				negative_derivative.append(d)

		cmpt += 1

	max_gradient = max(derivative_array)
	min_gradient = min(derivative_array)

	mean_pos_derivative = numpy.mean(postive_derivative)
	mean_neg_derivative = numpy.mean(negative_derivative)

	amplitude = float(max_y) - float(mean_first_range)

	line_to_write = str(data)+","+str(max_y)+","+str(mean_first_range)+","+str(max_gradient)+","+str(min_gradient)+","+str(mean_pos_derivative)+","+str(mean_neg_derivative)+","+str(amplitude)+"\n"
	result_file.write(line_to_write) 
	
	input_data.close()
result_file.close()

print "[*] => Done"
