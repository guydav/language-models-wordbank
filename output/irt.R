args = commandArgs(trailingOnly=TRUE)

#https://www.r-bloggers.com/2015/09/passing-arguments-to-an-r-script-from-command-lines/
# test if there is at least one argument: if not, return an error
if (length(args) != 4) {
	stop("Four arguments must be supplied (responses_file, model_names_file, item_parameters_output_file, ability_output_file)", call.=FALSE)
} #else if (length(args)==2) {
	# default output file
	#args[2] = "out.txt"
#}

library(mirt) #make mirt package available
library(tidyverse)

responses <- read.table(args[1], header = TRUE)
print("Responses file is read.")
model_names <- read.table(args[2], header = FALSE)
print("Model names file is read.")
names(model_names) <- c("name")
default_model_names <- data.frame(c("all-1", "all-0")) 
names(default_model_names) <- c("name")                 

#Adding observations using rbind() function  
full_model_names <- rbind(model_names, default_model_names)

init_theta <- data.frame(1:nrow(full_model_names))
names(init_theta) <- c("theta")
scores <- data.frame (
		theta = init_theta,
		model_names = full_model_names
)

#one factor, 2PL default item types (2PL)
#Page 116 of the documentation
#model <- 'F = 1-587
#		CONSTRAIN = (1-587, a1)'

#Page 125, 127 of the documentation
lognormal_prior <- 'F = 1-587
PRIOR = (1-587, a1, lnorm, 0, 1)' 
lognormal_model <- mirt.model(lognormal_prior)
#twoPL_run10_lognormal_prior <- mirt(lang_acq_run10_all_words_tsv, lognormal_model)

#RUN_16 has only 586 words used because of ignoring 'babysitter'
normal_prior <- 'F = 1-587
PRIOR = (1-587, a1, norm, 2.6, 1)' 
normal_model <- mirt.model(normal_prior)
#technical_parameters <- 'NCYCLES = 5'
irt_model <- mirt(responses, 1)

irt_parameters <- coef(irt_model, IRTpars = TRUE, simplify=TRUE)
#irt_parameters <- coef(irt_model)
#irt_parameters.to_tsv(args[3])
#https://stackoverflow.com/questions/17108191/how-to-export-proper-tsv/17108345
#write.table(irt_parameters, file=args[3], quote=FALSE, sep='\t', col.names = NA)
#print("Item parameters file is written.")

#original IRT metric for all items can be obtained using
#coef(irt_parameters, IRTpars = TRUE, simplify=TRUE)

#Page 59? of Documentation
#from Example_07.R
# basic fscores inputs using EAP estimator
#fscores(twoPL_run10)
#shows for each individual. If full.scores = FALSE, shows for each unique response pattern
thetas <- fscores(irt_model, full.scores=TRUE)
scores$theta <- thetas

#prepare data to write
scores <- scores %>% as_tibble()
irt_parameters <- as.data.frame(irt_parameters) %>% rownames_to_column(var = "word") %>% as_tibble()

#scores.to_tsv(args[4])
write.table(scores, file=args[4], quote=FALSE, sep='\t', col.names = NA)
print("Ability parameters file is written.")

write.table(irt_parameters, file=args[3], quote=FALSE, sep='\t')
print("Item parameters file is written.")