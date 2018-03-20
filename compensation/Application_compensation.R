## loading packages ##
suppressPackageStartupMessages(
{
  #library(xtable)	
  #library(flowBeads)
  library(flowCore)
  #library(flowStats)
  #library(flowViz)
  })

## convert string to lowercase excepted the first letter
convert_str <- function (str) {
  paste(toupper(substring(str, 1, 1)), tolower(substring(str, 2)), sep = "")
}

compensation <-function(file_path,comppath){
	flowDataPath = file_path
	flowCompPath = comppath
	compempty.mat = as.matrix(read.table("empty_matrix_comp.csv",header=TRUE,sep = ",",check.names = FALSE))

	flowData <-read.FCS(flowDataPath,transformation = FALSE, alter.names = FALSE,emptyValue=FALSE)
	names.flow <- colnames(flowData)[7:14]

	## Compensate ##s
	comp.mat = as.matrix(read.table(flowCompPath,header=TRUE,sep = "\t",check.names = FALSE)[,3:10])
	comp.mat=t(comp.mat/100)
	colnames(comp.mat) <- names.flow
	rownames(comp.mat) <- NULL
	format(comp.mat, scientific=TRUE)
	#compe <- compensation(comp.mat)
	flow_comp <- compensate(flowData ,comp.mat)
	flow_comp
	description(flow_comp)$`$SPILLOVER`
	description(flow_comp)$`$SPILLOVER`<- compempty.mat
	flow_comp
	description(flow_comp)
	filename = paste(file_path,"comp.fcs",sep = "_")
	write.FCS(flow_comp,filename , delimiter = "\\")
}





#### END FUNCTION ####

args <- commandArgs(trailingOnly = TRUE)

fcsPath<-args[1]
COMPPath<-args[2]
comppath<-"/home/glorfindel/Spellcraft/SIDEQUEST/compensation/choucroute_is_fun.txt"
file_path<-"/home/glorfindel/Spellcraft/SIDEQUEST/compensation/Panel_5_32152231_DRFZ_CANTO2_22JUL2016_22JUL2016.fcs_intra.fcs"
compensation(fcsPath,COMPPath)



