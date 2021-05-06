library(mirt) #make mirt package available

lang_acq_run08 <- read.csv("RUN_08_scores_for_R.csv", header = TRUE)
lang_acq_run09 <- read.csv("RUN_09_100words_scores_for_R.csv", header = TRUE)
lang_acq_run09_tsv <- read.table("RUN_09_100words_scores_for_R.tsv", header = TRUE)
lang_acq_run10_all_words_tsv <- read.table("RUN_10_all_words_scores_for_R.tsv", header = TRUE)
lang_acq_run16_29_models_all_words_tsv <- read.table("RUN_16_GPU_29_models_all_words_scores_for_R.tsv", header = TRUE)


#one factor, 2PL default item types (2PL)
#The second argument is the no of factors (we give 1 as we assume only one ability/latent factor)
twoPL_run08 <- mirt(lang_acq_run08, 1)
twoPL_run09 <- mirt(lang_acq_run09, 1)
twoPL_run10 <- mirt(lang_acq_run10_all_words_tsv, 1)
#Page 116 of the documentation
model <- 'F = 1-587
		CONSTRAIN = (1-587, a1)'
twoPL_run10_equal_slopes <- mirt(lang_acq_run10_all_words_tsv, model)
#Page 116 of the documentation
twoPL_run10_exploratory_2factor <- mirt(lang_acq_run10_all_words_tsv, 2)

#Page 125, 127 of the documentation
lognormal_prior <- 'F = 1-587
PRIOR = (1-587, a1, lnorm, 0, 1)' 
lognormal_model <- mirt.model(lognormal_prior)
twoPL_run10_lognormal_prior <- mirt(lang_acq_run10_all_words_tsv, lognormal_model)

#FIGURE OUT why RUN_16 has only 586 words used.
normal_prior <- 'F = 1-586
PRIOR = (1-586, a1, norm, 2.6, 1)' 
normal_model <- mirt.model(normal_prior)
twoPL_run16_29_models_all_words_normal_prior <- mirt(lang_acq_run16_29_models_all_words_tsv, normal_model)

#help('coef-method')

#coef(twoPL_run08) #in slope-intercept form, use b = -d/a to obtain traditional metric
#(b2 <- -0.808/1.081)

#original IRT metric for all items can be obtained using
coef(twoPL_run08, IRTpars = TRUE, simplify=TRUE)
coef_run16 <- coef(twoPL_run16_29_models_all_words_normal_prior)

#imposed tracelines
help('plot-method')
plot(twoPL_run09, type = 'trace')
plot(twoPL_run09, type = 'trace', auto.key = FALSE) #without legend

#???
plot(twoPL_run10, type = 'info')

#Page 59? of Documentation
#from Example_07.R
help(fscores)
# basic fscores inputs using EAP estimator
fscores(twoPL_run10)
#shows for each individual. If full.scores = FALSE, shows for each unique response pattern
fscores(twoPL_run10, full.scores=TRUE)