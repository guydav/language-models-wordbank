library(mirt) #make mirt package available

lang_acq_run08 <- read.csv("RUN_08_scores_for_R.csv", header = TRUE)
lang_acq_run09 <- read.csv("RUN_09_100words_scores_for_R.csv", header = TRUE)
lang_acq_run09_tsv <- read.table("RUN_09_100words_scores_for_R.tsv", header = TRUE)


#one factor, 2PL default item types (2PL)
twoPL_run08 <- mirt(lang_acq_run08, 1)
twoPL_run09 <- mirt(lang_acq_run09, 1)


#help('coef-method')

#coef(twoPL_run08) #in slope-intercept form, use b = -d/a to obtain traditional metric
#(b2 <- -0.808/1.081)

#original IRT metric for all items can be obtained using
coef(twoPL_run08, IRTpars = TRUE, simplify=TRUE)

#imposed tracelines
help('plot-method')
plot(twoPL_run09, type = 'trace')
plot(twoPL_run09, type = 'trace', auto.key = FALSE) #without legend