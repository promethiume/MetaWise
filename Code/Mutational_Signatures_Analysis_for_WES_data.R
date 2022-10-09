########################
## Required libraries ##
########################
setwd('cBioPortal_datasets/')
library(maftools)
library(sigminer)

#####################
## Data processing ##
#####################
## process TCGA mutation data
tcga_cohorts <- list.files('./', pattern = 'tcga')
tcga_mut_list <- lapply(tcga_cohorts, function(x){
  mut = read.delim(paste(x, '/', 'data_mutations.txt', sep = ''), comment.char = '#')
  mut = mut[, c('Hugo_Symbol', 'Chromosome', 'Start_Position', 'End_Position', 'Strand', 'Variant_Classification', 'Variant_Type',
                'Reference_Allele', 'Tumor_Seq_Allele1', 'Tumor_Seq_Allele2', 'Tumor_Sample_Barcode')]
  return(mut)
})

tcga_mut_df = do.call(rbind, tcga_mut_list)

## process metastatic cohorts mutation data
met_cohorts <- list.files('metastatic_WES/')
met_cohorts
met_mut_list <- lapply(met_cohorts, function(x){
  mut = read.delim(paste('metastatic_WES/', x, '/data_mutations.txt', sep = ''), comment.char = '#')
  mut = mut[, c('Hugo_Symbol', 'Chromosome', 'Start_Position', 'End_Position', 'Strand', 'Variant_Classification', 'Variant_Type',
                'Reference_Allele', 'Tumor_Seq_Allele1', 'Tumor_Seq_Allele2', 'Tumor_Sample_Barcode')]
  return(mut)
})

met_mut_df = do.call(rbind, met_mut_list)

## combine primary and metastasis mutation data
wes_mut_df = rbind(tcga_mut_df, met_mut_df)

## classify WES mutation data into datasets with and without non-coding mutations
vc = names(table(wes_mut_df$Variant_Classification)) ## extract all types of variant classification

coding_vc = c('Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Missense_Mutation', 
              'Nonsense_Mutation', 'Silent', 'Splice_Site', 'Translation_Start_Site', 'Nonstop_Mutation') ## vc for coding regions


wes_maf = read.maf(wes_mut_df, vc_nonSyn = vc)
wes_maf_wo_nc = read.maf(wes_mut_df[wes_mut_df$Variant_Classification %in% coding_vc, ], vc_nonSyn = coding_vc)

#### tally mutation types
mt_tally <- sig_tally(
  wes_maf,
  method = 'S',
  mode = 'ALL',
  ref_genome = 'BSgenome.Hsapiens.UCSC.hg19',
  useSyn = TRUE,
  cores = 6
)

mt_tally_wo_nc <- sig_tally(
  wes_maf_wo_nc,
  method = 'S',
  mode = 'ALL',
  ref_genome = 'BSgenome.Hsapiens.UCSC.hg19',
  useSyn = TRUE,
  cores = 6
)

#### fitting to COSMIC V3.3 mutational signatures
v3.3_sbs = read.delim('~/ann/COSMIC_Mutational_Signaturs_V3.3/COSMIC_v3.3_SBS_GRCh37.txt', row.names = 1)
v3.3_dbs = read.delim('~/ann/COSMIC_Mutational_Signaturs_V3.3/COSMIC_v3.3_DBS_GRCh37.txt', row.names = 1)
v3.3_id = read.delim('~/ann/COSMIC_Mutational_Signaturs_V3.3/COSMIC_v3.3_ID_GRCh37.txt', row.names = 1)

## fit sbs, dbs and id matrices to COSMIC v3.3 reference signatures
sbs_fit <- sig_fit(t(mt_tally$SBS_96), sig = v3.3_sbs)
dbs_fit <- sig_fit(t(mt_tally$DBS_78), sig = v3.3_dbs)
id_fit <- sig_fit(t(mt_tally$ID_83), sig = v3.3_id)

wes_fit <- rbind(sbs_fit, dbs_fit, id_fit)

sbs_fit_wo_nc <- sig_fit(t(mt_tally_wo_nc$SBS_96), sig = v3.3_sbs)
dbs_fit_wo_nc <- sig_fit(t(mt_tally_wo_nc$DBS_78), sig = v3.3_dbs)
id_fit_wo_nc <- sig_fit(t(mt_tally_wo_nc$ID_83), sig = v3.3_id)

wes_fit_wo_nc <- rbind(sbs_fit_wo_nc, dbs_fit_wo_nc, id_fit_wo_nc)

#### Denovo extraction of mutational signatures
sigprofiler_extract(nmf_matrix = mt_tally$SBS_96, output = 'WES_sp_sig30_results', range = 30,
                    genome_build = 'hg19', cores = 10)


sigprofiler_extract(nmf_matrix = mt_tally_wo_nc$SBS_96, output = 'WES_sp_sig30_results_without_non-coding', range = 30,
                    genome_build = 'hg19', cores = 10)

#### Clinical data processing
tcga_clini_list <- lapply(tcga_cohorts, function(x){
  clini = read.delim(paste(x, '/', 'data_clinical_sample.txt', sep = ''), comment.char = '#')
  clini$Cohort = gsub('_tcga_pan_can_atlas_2018', '', x)
  return(clini)
})

tcga_clini_df = do.call(rbind, tcga_clini_list)
tcga_clini_primary = tcga_clini_df[tcga_clini_df$SAMPLE_TYPE == 'Primary', ]
tcga_clini_metastasis = tcga_clini_df[tcga_clini_df$SAMPLE_TYPE == 'Metastasis', ]

samp_clini_df = tcga_clini_primary[, c('SAMPLE_ID', 'ONCOTREE_CODE', 'SAMPLE_TYPE')]
samp_clini_df = rbind(samp_clini_df, tcga_clini_metastasis[, c('SAMPLE_ID', 'ONCOTREE_CODE', 'SAMPLE_TYPE')])

list.files('metastatic_WES/')

met_brca_igr_2015_cli = read.delim('metastatic_WES/brca_igr_2015/data_clinical_sample.txt', comment.char = '#')
met_brca_igr_2015_cli = met_brca_igr_2015_cli[, c('SAMPLE_ID', 'ONCOTREE_CODE', 'SAMPLE_TYPE')]

mel_dfci_cli = read.delim('metastatic_WES/mel_dfci_2019/data_clinical_sample.txt', comment.char = '#')
mel_dfci_cli = mel_dfci_cli[, c('SAMPLE_ID', 'ONCOTREE_CODE')]
mel_dfci_cli$SAMPLE_TYPE = 'Metastasis'

met_500_cli = read.delim('metastatic_WES/metastatic_solid_tumors_mich_2017/data_clinical_sample.txt', comment.char = '#')
met_500_cli = met_500_cli[, c('SAMPLE_ID', 'ONCOTREE_CODE', 'SAMPLE_TYPE')]

nepc_cli = read.delim('metastatic_WES/nepc_wcm_2016/data_clinical_sample.txt', comment.char = '#')
table(nepc_cli$SAMPLE_ID %in% met_mut_df$Tumor_Sample_Barcode)
nepc_cli = nepc_cli[, c('SAMPLE_ID', 'ONCOTREE_CODE')]
nepc_cli$SAMPLE_TYPE = 'Metastasis'

prad_cli = read.delim('metastatic_WES/prad_su2c_2015/data_clinical_sample.txt', comment.char = '#')
prad_cli = prad_cli[, c('SAMPLE_ID', 'ONCOTREE_CODE')]
prad_cli$SAMPLE_TYPE = 'Metastasis'

skcm_cli = read.delim('metastatic_WES/skcm_dfci_2015/data_clinical_sample.txt', comment.char = '#')
skcm_cli = skcm_cli[, c('SAMPLE_ID', 'ONCOTREE_CODE')]
skcm_cli$SAMPLE_TYPE = 'Metastasis'


met_clini = rbind(met_brca_igr_2015_cli, mel_dfci_cli, met_500_cli, nepc_cli, prad_cli, skcm_cli)

wes_samp_clini = rbind(samp_clini_df, met_clini)

wes_samp_clini = wes_samp_clini[wes_samp_clini$SAMPLE_ID %in% wes_mut_df$Tumor_Sample_Barcode, ]
write.table(wes_samp_clini, file = 'WES_clinical_sample.txt', row.names=F, col.names=T, sep = '\t', quote=F)

wes_fit = wes_fit[, colnames(wes_fit) %in% wes_samp_clini$SAMPLE_ID]
wes_fit_wo_nc = wes_fit_wo_nc[, colnames(wes_fit_wo_nc) %in% wes_samp_clini$SAMPLE_ID]

write.table(wes_fit, file = 'WES_fit-COSMIC_MutSigs_Exposure-Mat.txt', row.names = T, col.names = T, sep = '\t', quote = F)
write.table(wes_fit_wo_nc, file = 'WES_fit-COSMIC_MutSigs_Exposure-Mat_without_non-coding.txt', row.names = T, col.names = T, sep = '\t', quote = F)


sig30 = read.delim('WES_sp_sig30_results/SBS96/All_Solutions/SBS96_30_Signatures/Activities/SBS96_S30_NMF_Activities.txt')
sig30 = sig30[sig30$Samples %in% wes_samp_clini$SAMPLE_ID, ]
sig30_wo_nc = read.delim('WES_sp_sig30_results_without_non-coding/SBS96/All_Solutions/SBS96_30_Signatures/Activities/SBS96_S30_NMF_Activities.txt')
sig30_wo_nc = sig30_wo_nc[sig30_wo_nc$Samples %in% wes_samp_clini$SAMPLE_ID, ]

rownames(sig30) = sig30$Samples
rownames(sig30_wo_nc) = sig30_wo_nc$Samples

sig30 = sig30[, -1]
sig30_wo_nc = sig30_wo_nc[, -1]
 
s30 = t(sig30)
s30_wo_nc = t(sig30_wo_nc)

write.table(s30, file = 'WES_sig30_Exposure-Mat.txt', row.names=T, col.names=T, sep = '\t', quote=F)
write.table(s30_wo_nc, file = 'WES_sig30_Exposure-Mat_without_non-coding.txt', row.names=T, col.names = T, sep = '\t', quote = F)
