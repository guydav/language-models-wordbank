library(mirt) #make mirt package available

lang_acq_run16_29_models_all_words_tsv <- read.table("RUN_16_GPU_29_models_all_words_scores_for_R.tsv", header = TRUE)

#one factor, 2PL default item types (2PL)
twoPL_run16_29_models_all_words <- mirt(lang_acq_run16_29_models_all_words_tsv, 1)

#help('coef-method')

#coef(twoPL_run08) #in slope-intercept form, use b = -d/a to obtain traditional metric
#(b2 <- -0.808/1.081)

#original IRT metric for all items can be obtained using
coef_run16 <- coef(twoPL_run16_29_models_all_words, IRTpars = TRUE, simplify=TRUE)

#plot(twoPL_run16_29_models_all_words, type = 'trace')
#plot(twoPL_run16_29_models_all_words, type = 'trace', auto.key = FALSE) #without legend

#???
#plot(twoPL_run10, type = 'info')

#Page 59? of Documentation
#from Example_07.R
#help(fscores)
# basic fscores inputs using EAP estimator
fscores(twoPL_run16_29_models_all_words)
#shows for each individual. If full.scores = FALSE, shows for each unique response pattern
fscores(twoPL_run16_29_models_all_words, full.scores=TRUE)




#Wordbank data
load("../../wordbank-book/data/psychometrics/eng_ws_raw_data.Rds")
#install.packages('magrittr')
#install.packages('dplyr')
#install.packages('purrr')
#OR
#install.packages("tidyverse")#https://www.tidyverse.org/packages/
library(magrittr)
library(dplyr)
library(purrr)
library(ggplot2)
library(tibble)

#https://www.datacamp.com/community/tutorials/pipe-r-tutorial

coef_run16_tibble <- as_tibble(coef_run16)
#https://github.com/langcog/wordbank-book/blob/master/004-psychometrics.Rmd
thetas <- seq(-6,6,.1)
irt4pl <- function(a, d, g, u, theta = seq(-6,6,.1)) {
	p = g + (u - g) * boot::inv.logit(a * (theta + d))
	return(p)
}
irt2pl <- function(a, d, theta = seq(-6,6,.1)) {
	p = boot::inv.logit(a * (theta + d))
	return(p)
}
examples <- c("table","mommy*","trash","yesterday")
iccs <- coef_run16_tibble %>%
		filter(definition %in% examples) %>%
		split(.$definition) %>%
		map_df(function(d) {
					return(data_frame(definition = d$definition,
									theta = thetas, 
									p = irt2pl(d$a1, d$d, thetas)))
				})
ggplot(iccs,  
				aes(x = theta, y = p)) + 
		geom_line() + 
		facet_wrap(~definition) + 
		xlab("Ability") + 
		ylab("Probability of production")

ggplot(coef_run16,  
				aes(x = a1, y = -d)) + 
		geom_point(alpha = .3) + 
		ggrepel::geom_text_repel(data = filter(coef_run16, 
						-d < -3.8 | -d > 5.3 | a1 > 4 | a1 < 1), 
				aes(label = definition), size = 3) + 
		xlab("Discrimination") + 
		ylab("Difficulty")

