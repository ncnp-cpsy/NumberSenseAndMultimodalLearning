fname_anovakun = "./anovakun_485.txt"
synthesized_data = "./synthesized.csv"
source(fname_anovakun)

cols <- c(
    "magnitude_avg",
    # magnitude_log_mean_avg,
    "magnitude_exp_mean_avg",
    "magnitude_pow_mean_avg",
    "magnitude_logmin_mean_avg"
)
df_original <- read.csv(synthesized_data)

cat("Dimension of original data:", dim(df_original))
head(df_original)

for (target_modality in c(0, 1)) {
    print("Target modality")
    print(target_modality)

    df_anovakun <- df_original[
    (df_original['model_name'] == 'MMVAE_CMNIST_OSCN' & df_original['target_modality'] == target_modality), cols
    ]

    cat("\n\nDimension of anovakun data:", dim(df_anovakun))
    print(df_anovakun)
    print(psych::describe(df_anovakun))
    # print(psych::describeBy(df_anovakun, df_anovakun[, ]))

    rslt_anovakun <- anovakun(df_anovakun, "sA", 4)
    print(rslt_anovakun)
}
