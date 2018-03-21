import FlowCal
import matplotlib.pyplot as plt
import glob
import random
import numpy

test_file = 'Panel_5_32152231_DRFZ_CANTO2_22JUL2016_22JUL2016.fcs_intra.fcs'
test_file_2 = "/home/elwin/PANEL_5/PANEL_5/UBO/Panel_5_32170096_UBO_NAVIOS_19JUN2017_19JUN2017.LMD_intra.fcs"


## open file
s = FlowCal.io.FCSData('Panel_5_32152231_DRFZ_CANTO2_22JUL2016_22JUL2016.fcs_intra.fcs')

## biplot
print s.channels
FlowCal.plot.density2d(s, channels=['FSC-A', 'SSC-A'], mode='scatter')
#plt.show()


def extract_c_matrix(filename):
	##
	## Try something
	##
	## only work on 8 channel for now, use it as a delimiter
	## 

	matrix = {}

	data = open(filename, "r")
	cmpt = 0
	for line in data:
		if(cmpt == 0 ):
						
			line = line.split("\\8,")
			line = line[1]
			line = line.split("\\")
			line = line[0]

			line_in_array = line.split(",")
			vector = line_in_array[:8]
			scalars = line_in_array[8:]

			vector_list = []
			for x in xrange(0,8):
				v = scalars[x*8:(x*8)+8]
				vector_list.append(v)

			cmpt1 = 0
			for channel in vector:
				matrix[channel] = {}
				v = vector_list[cmpt1]
				cmpt1 += 1
				cmpt2 = 0
				for channel2 in vector:
					matrix[channel][channel2] = v[cmpt2]
					cmpt2 += 1
		cmpt +=1
	data.close()

	return matrix


def generate_random_matrix():
	##
	## IN PROGRESS
	##

	matrix = {}
	channels = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]
	
	for c1 in channels:
		matrix[c1] = {}
		for c2 in channels:
			matrix[c1][c2] = "NA"

	for c1 in channels:
		for c2 in channels:
			if(c1 == c2):
				matrix[c1][c2] = 1
			elif(matrix[c2][c1] != "NA"):
				matrix[c1][c2] = matrix[c2][c1] 
			else:
				matrix[c1][c2] = random.random()

	return matrix



def write_matrix_in_line(matrix):
	##
	## IN PROGRESS
	##

	matrix_in_line = ""
	channels = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]
	for c in channels:
		matrix_in_line += str(c)+","
	for c1 in channels:
		for c2 in channels:
			matrix_in_line += str(matrix[c1][c2])+","
	matrix_in_line = matrix_in_line[:-1]

	return matrix_in_line



def edit_file(fcs_file):
	##
	## IN PROGRESS
	##


	original_matrix = extract_c_matrix(fcs_file)
	original_matrix = write_matrix_in_line(original_matrix)
	new_matrix = generate_random_matrix()
	new_matrix = write_matrix_in_line(new_matrix)

	new_matrix = original_matrix.replace("0.0139005340407433", "0.01390053")


	print original_matrix
	print len(original_matrix.split(","))
	print "-------------"
	print new_matrix
	print len(new_matrix.split(","))

	output_file = open("truc.fcs", "w")
	input_file = open(fcs_file, "r")
	for line in input_file:
		if original_matrix in line:
			line = line.replace(original_matrix, new_matrix)
		output_file.write(line)
	input_file.close()
	output_file.close()





def generate_random_compensation_matrix_file(output_file):
	##
	## IN PROGRESS
	##


	matrix = {}
	channels = ["FITC.A","PE.A","PC5.5.A","PC7.A","APC.A","APC.AF750.A","PB.A","KO.A"]
	
	for c1 in channels:
		matrix[c1] = {}
		for c2 in channels:
			matrix[c1][c2] = "NA"


	matrix_file = open(output_file, "w")

	header = "ravioli\travioli\tFITC.A\tPE.A\tPC5.5.A\tPC7.A\tAPC.A\tAPC.AF750.A\tPB.A\tKO.A\n"
	matrix_file.write(header)

	for x in xrange(0,8):
		vector_in_line = ""
		for y in xrange(0,10):
			
			if(y < 2):
				vector_in_line += "choucroute\t"
			elif(y-2 == x):
				vector_in_line += "100\t"
				matrix[channels[x]][channels[y-2]] = 100
			else:
				scalar = random.uniform(0.0,50.0)
				vector_in_line += str(scalar)+"\t"

	
		vector_in_line = vector_in_line[:-1]
		matrix_file.write(vector_in_line+"\n")



	matrix_file.close()



def load_matrix_from_compensation_file(load_file):
	##
	## => Create a matrix from a save_file,
	## cast the matrix into an numpy array
	## return the matrix
	##

	matrix = []
	input_data = open(load_file, "r")
	cmpt = 0
	for line in input_data:
		
		if(cmpt > 0):
			vector = []
			line = line.replace("\n", "")
			line_in_array = line.split("\t")
			
			index = 0
			for scalar in line_in_array:
				if(index > 1):

					vector.append(float(scalar))
				index += 1

			matrix.append(vector)

		cmpt += 1

	input_data.close()

	matrix = numpy.array(matrix)

	return matrix




def sex_matrix(parent_matrix_1, parent_matrix_2):
	##
	## => A very simple reproduction function, create a new matrix
	##    and for each scalar run the dice to know if it should come
	##    from parent 1 or parent 2, check if variable not already in
	##    matrix
	##
	## => can return False Matrix
	##

	child_matrix = numpy.zeros(shape=(len(parent_matrix_1),len(parent_matrix_1[0])))
	scalar_in_child = []
	errors_cmpt = 0
	errors_position_list = []

	for x in xrange(0,len(child_matrix)):
		for y in xrange(0, len(child_matrix[0])):

			## roll the dices
			random_choice = random.randint(0,100)

			if(random_choice > 50):
				scalar_to_add = parent_matrix_1[x][y]
				if(scalar_to_add not in scalar_in_child):
					child_matrix[x][y] = parent_matrix_1[x][y]
					scalar_in_child.append(scalar_to_add)
				else:
					if(parent_matrix_2[x][y] not in scalar_in_child):
						child_matrix[x][y] = parent_matrix_2[x][y]
						scalar_in_child.append(parent_matrix_2[x][y])
					else:
						child_matrix[x][y] = -1
						errors_position_list.append([x,y])
						errors_cmpt += 1
			else:
				scalar_to_add = parent_matrix_2[x][y]
				if(scalar_to_add not in scalar_in_child):
					child_matrix[x][y] = parent_matrix_2[x][y]
					scalar_in_child.append(scalar_to_add)
				else:
					if(parent_matrix_1[x][y]):
						child_matrix[x][y] = parent_matrix_1[x][y]
						scalar_in_child.append(parent_matrix_1[x][y])
					else:
						child_matrix[x][y] = -1
						errors_position_list.append([x,y])
						errors_cmpt += 1

	return child_matrix





import os
import shutil

def generate_images_from_fcs():
	##
	## IN PROGRESS
	##

	## training dataset

	## uncompsated
	## list input fcs file
	input_files = glob.glob("data/fcs/raw/*.fcs")

	## run R script
	for fcs_file in input_files:

		print "[INFO] - Generate image for "+str(fcs_file)
		os.system("Rscript print_plot.R "+str(fcs_file))

	## deal with generated images
	output_files = glob.glob("data/fcs/raw/*.png")
	for png_file in output_files:

		print "[INFO] - Moving image "+str(png_file)
		dist = png_file.split("/")
		dist = dist[-1]
		dist = "data/images/raw/"+str(dist)

		shutil.copy(png_file, dist)

	## compensated
	## list input fcs file
	input_files = glob.glob("data/fcs/compensated/*.fcs")

	## run R script
	for fcs_file in input_files:
		print "[INFO] - Generate image for "+str(fcs_file)
		os.system("Rscript print_plot.R "+str(fcs_file))

	## deal with generated images
	output_files = glob.glob("data/fcs/compensated/*.png")
	for png_file in output_files:

		print "[INFO] - Moving image "+str(png_file)
		dist = png_file.split("/")
		dist = dist[-1]
		dist = "data/images/compensated/"+str(dist)

		shutil.copy(png_file, dist)





#generate_images_from_fcs()


#machin = load_matrix_from_compensation_file("choucroute_is_fun.txt")
#truc = load_matrix_from_compensation_file("Panel_1_UBO_32151130.txt")

#bidule = sex_matrix(machin, truc)
#print bidule


"""
generate_random_compensation_matrix_file("choucroute_is_fun.txt")


#truc = generate_random_matrix()
#write_matrix_in_line(truc)
#edit_file(test_file)

## open file
s = FlowCal.io.FCSData('Panel_5_32152231_DRFZ_CANTO2_22JUL2016_22JUL2016.fcs_intra.fcs')
FlowCal.plot.density2d(s, channels=['APC.A', 'APC.AF750.A'], mode='scatter')
plt.show()
plt.close()

s = FlowCal.io.FCSData('Panel_5_32152231_DRFZ_CANTO2_22JUL2016_22JUL2016.fcs_intra.fcs_comp.fcs')
FlowCal.plot.density2d(s, channels=['APC.A', 'APC.AF750.A'], mode='scatter')
try:
	plt.show()
	plt.close()
except:
	print "tardis"
"""




"""
for test_file in glob.glob("/home/elwin/PANEL_5/PANEL_5/UBO/*.fcs"):
	#extract_c_matrix(test_file)

	edit_file(test_file)
"""