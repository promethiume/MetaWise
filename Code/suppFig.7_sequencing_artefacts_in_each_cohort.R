
##############################################################################
## Analysis of sequencing artefacts signatures distributions in the cohorts ##
##############################################################################
library(reshape2)
library(pheatmap)

setwd('~/1.Metastasis_Project/1.Public_Data/TCGA_WES/cBioPortal')

mut_sigs_wes_107 <- read.delim('~/1.Metastasis_Project/2.MutSigs_Exposure_Mats/WES/WES_fit-COSMIC_sig107_MutSigs_Exposure-Mat.txt', 
                               stringsAsFactors = FALSE, check.names = FALSE)
wes_clini <- read.delim('~/1.Metastasis_Project/2.MutSigs_Exposure_Mats/WES/WES_clinical_sample.txt')

tcga_cohorts = list.files('./', pattern = 'tcga')

tcga_clini <- lapply(tcga_cohorts, function(x){
  clini = read.delim(paste(x, '/', 'data_clinical_sample.txt', sep = ''), comment.char = '#')
  clini$Cohort = gsub('_tcga_pan_can_atlas_2018', '', x)
  clini$Cohort = paste('TCGA-', toupper(clini$Cohort), sep = '')
  return(clini)
})

tcga_clini = do.call(rbind, tcga_clini)
head(tcga_clini)


list.files('metastatic_WES/')
## Process clinical information from metastatic cohorts
met_brca_igr_2015_cli = read.delim('metastatic_WES/brca_igr_2015/data_clinical_sample.txt', comment.char = '#')
met_brca_igr_2015_cli$Cohort = 'brca_igr_2015'

met_brca_mbc_cli = read.delim('metastatic_WES/brca_mbcproject_wagle_2017/data_clinical_sample.txt', comment.char = '#')
met_brca_mbc_cli$Cohort = 'brca_mbcproject_wagle_2017'

mel_dfci_cli = read.delim('metastatic_WES/mel_dfci_2019/data_clinical_sample.txt', comment.char = '#')
mel_dfci_cli$Cohort = 'mel_dfci_2019'

met_500_cli = read.delim('metastatic_WES/metastatic_solid_tumors_mich_2017/data_clinical_sample.txt', comment.char = '#')
met_500_cli$Cohort = 'met500'

nepc_cli = read.delim('metastatic_WES/nepc_wcm_2016/data_clinical_sample.txt', comment.char = '#')
nepc_cli$Cohort = 'nepc_wcm_2016'

prad_cli = read.delim('metastatic_WES/prad_su2c_2015/data_clinical_sample.txt', comment.char = '#')
prad_cli$Cohort = 'prad_su2c_2015'

skcm_cli = read.delim('metastatic_WES/skcm_dfci_2015/data_clinical_sample.txt', comment.char = '#')
skcm_cli$Cohort = 'skcm_dfci_2015'


wes_samp_clini = rbind(tcga_clini[, c('SAMPLE_ID', 'Cohort')], met_brca_igr_2015_cli[, c('SAMPLE_ID', 'Cohort')], met_brca_mbc_cli[, c('SAMPLE_ID', 'Cohort')],
                       mel_dfci_cli[, c('SAMPLE_ID', 'Cohort')], met_500_cli[, c('SAMPLE_ID', 'Cohort')], nepc_cli[, c('SAMPLE_ID', 'Cohort')], prad_cli[, c('SAMPLE_ID', 'Cohort')],
                       skcm_cli[, c('SAMPLE_ID', 'Cohort')])
head(wes_samp_clini)

## Calculate relative enrichment of each sequencing artefacts in each cohort
seq_artefacts = c('SBS27', 'SBS43', 'SBS45', 'SBS46', 'SBS47', 'SBS48', 'SBS49', 
                  'SBS50', 'SBS51', 'SBS52', 'SBS53', 'SBS54', 'SBS55', 'SBS56', 
                  'SBS57', 'SBS58', 'SBS59', 'SBS60', 'SBS95')

ms_seq_artefacts = mut_sigs_wes_107[rownames(mut_sigs_wes_107) %in% seq_artefacts, ]

sigs_percent_in_cohort_list = lapply(unique(wes_samp_clini$Cohort), function(x, exp_cutoff = 0){
  ana_cohort = x
  df = ms_seq_artefacts[, colnames(ms_seq_artefacts) %in% wes_samp_clini$SAMPLE_ID[wes_samp_clini$Cohort == ana_cohort]]
  samp_count = dim(df)[2]
  df0 = df
  if (exp_cutoff > 0){
    df0[df0 >= exp_cutoff] = 1
    df0[df0 < exp_cutoff] = 0
  }else{
    df0[df0 > exp_cutoff] = 1
  }
  
  st = rowSums(df0) / samp_count * 100
  st= as.data.frame(st)
  st$cohort = ana_cohort
  st$sigs = rownames(st)
  return(st)
})

sigs_percent_in_cohort_df = do.call(rbind, sigs_percent_in_cohort_list)
sigs_percent_in_cohort_mat = dcast(sigs_percent_in_cohort_df, sigs ~ cohort, value.var = 'st')
sigs_percent_in_cohort_mat[is.na(sigs_percent_in_cohort_mat)] = 0
rownames(sigs_percent_in_cohort_mat) = sigs_percent_in_cohort_mat$sigs
sigs_percent_in_cohort_mat = sigs_percent_in_cohort_mat[, -1]

pdf('Percentage_of_samples_associated_with_possible_sequencing_artefacts_signatures_in_each_cohort.pdf', 14.82, 8.22)
print(pheatmap(sigs_percent_in_cohort_mat, angle_col = 45, cellwidth = 20, cellheight = 20, border_color = 'black', 
               main = 'Percentage of samples associated with possible sequencing artefacts signatures in each cohort'))
dev.off()
