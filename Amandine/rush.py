

##
## The idea is to split the matrix
## to gain max information in sub matrix
##

variable_to_index = {}
index_to_variable = {}
variable_to_value = {}

input_data = open("5_aza.csv", "r")
cmpt = 0
for line in input_data:

	## Drop windows stuff
	line = line.replace("\\r", "")
	line = line.replace("\\xef\\xbb\\xbf", "")

	if(cmpt == 0):

		line_in_array = line.split(";")
		
		index = 0
		for variable in line_in_array:
			
			## Drop windows stuff
			variable = variable.replace("\\r", "")
			variable = variable.replace("\\xef\\xbb\\xbf", "")
			variable_to_index[variable] = index
			index_to_variable[index] = variable
			variable_to_value[variable] = []
			index += 1

	else:

		line_in_array = line.split(";")
		index = 0
		for scalar in line_in_array:
			scalar = scalar.replace("\\r", "")
			scalar = scalar.replace("\\xef\\xbb\\xbf", "")
			variable_to_value[index_to_variable[index]].append(scalar)
			index += 1


	cmpt += 1
input_data.close()


## Get patient number for each variable where scalar is not NA
variable_to_valid_patient = {}
for variable in variable_to_value.keys():
	vector = variable_to_value[variable]
	variable_to_valid_patient[variable] = []

	index = 0
	for scalar in vector:
		if(scalar != "NA"):
			variable_to_valid_patient[variable].append(index)
		index += 1


print variable_to_valid_patient

## class list proximity



