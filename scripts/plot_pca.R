library(ggplot2)
library(ggrepel)
set.seed(1) # ggrepel reproducibility

sample_pops <- readr::read_tsv(snakemake@input$sample_pops, show_col_types = FALSE)
projections <- readr::read_tsv(snakemake@input$proj, show_col_types = FALSE) |>
  dplyr::inner_join(sample_pops, by = c("#IID" = "sample")) |>
  dplyr::mutate(pop = as.factor(pop))

variance <- readr::read_tsv(snakemake@input$variance, col_names = c("PC_var_explained"), show_col_types = FALSE)
outlier_threshold_sd <- snakemake@params$outlier_threshold_sd

projections <- dplyr::mutate(
  projections,
  # outlier if EITHER PC1 OR PC2 is more than threshold * SD away from the mean
  outlier = abs(PC1 - mean(PC1)) > outlier_threshold_sd * sd(PC1) |
    abs(PC2 - mean(PC2)) > outlier_threshold_sd * sd(PC2)
)

pca_plot <- ggplot(projections, aes(x = PC1, y = PC2)) +
  geom_point(aes(color = super_pop)) +
  geom_label_repel(
    data = \(d) dplyr::filter(d, outlier),
    aes(label = `#IID`)
  ) +
  scale_color_viridis_d() +
  labs(
    x = paste0("PC1 (", round(variance$PC_var_explained[1], 2), "%)"),
    y = paste0("PC2 (", round(variance$PC_var_explained[2], 2), "%)"),
    title = paste("PCA:", snakemake@wildcards$pop_1kg, snakemake@wildcards$chr)
  ) +
  theme_bw()

ggsave(
  snakemake@output$pca_plot,
  pca_plot,
  height = 8,
  width = 8
)