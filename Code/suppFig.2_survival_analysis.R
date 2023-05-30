## Load Primary samples predicted results by MetaWise-Pan model in test sets
pred_res = read.delim('Primary_samples_model_predicted_results.txt')

## Load clinical data of TCGA primary samples
tcga_cohorts <- list.files('~/1.Metastasis_Project/1.Public_Data/TCGA_WES/cBioPortal/', pattern = 'tcga', full.names = TRUE)

tcga_patient_clini_info <- lapply(tcga_cohorts, function(x){
  print(x)
  clini = read.delim(paste(x, '/', 'data_clinical_patient.txt', sep = ''), comment.char = '#')
  clini = clini[, c('PATIENT_ID', 'SUBTYPE', 'AGE', 'SEX', 'AJCC_PATHOLOGIC_TUMOR_STAGE', 'PATH_M_STAGE', 'PATH_N_STAGE', 'PATH_T_STAGE',
                    'OS_STATUS', 'OS_MONTHS', 'DSS_STATUS', 'DSS_MONTHS', 'DFS_STATUS', 'DFS_MONTHS', 'PFS_STATUS', 'PFS_MONTHS',
                    'NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT', 'RADIATION_THERAPY', 'WEIGHT', 'PERSON_NEOPLASM_CANCER_STATUS')]
  return(clini)
})

tcga_patient_clini_info = do.call(rbind, tcga_patient_clini_info)

## Retrieve clinical information for the Primary samples in test sets
pred_res_clini_info = tcga_patient_clini_info[match(pred_res$PATIENT_ID, tcga_patient_clini_info$PATIENT_ID), ]
pred_res = cbind(pred_res, pred_res_clini_info)

## Survival analysis of LMPS and HMPS groups
library(survminer)
library(survival)

pred_res$DSS_STATUS[pred_res$DSS_STATUS == '0:ALIVE OR DEAD TUMOR FREE'] = 0
pred_res$DSS_STATUS[pred_res$DSS_STATUS == '1:DEAD WITH TUMOR'] = 1
pred_res$DSS_STATUS = as.numeric(pred_res$DSS_STATUS)

sfit_dss = survfit(Surv(DSS_MONTHS, DSS_STATUS) ~ Group, data = pred_res)
pdf('DSS_survival.pdf', 5.25, 6.05)
print(ggsurvplot(sfit_dss, conf.int = F, pval = TRUE, risk.table = TRUE))
dev.off()