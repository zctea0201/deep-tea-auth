set.seed(2024)
pacman::p_load(tidyverse, plyr, mixOmics)
work_dir <- "D:/A10-Project/2022_TGY/LC_2021/"

# --- Data Loading and Preprocessing ---

data_file <- str_c(work_dir, "02.data_clearning/data_raw.csv")
data_all <- read_csv(data_file)

sample_file <- str_c(work_dir, "00.sample_info/sample_info.csv")
sample_info <- read.csv(sample_file) %>% 
  select(sample_id, group.aroma_season, group.ml_2024) %>%
  set_names("label", "class", "group") %>% 
  filter(class != "QC") # Exclude Quality Control samples

train_info <- sample_info %>% 
  filter(group == "train")
test_info <- sample_info %>% 
  filter(group == "test")

X <- read_csv(data_file) %>%
  dplyr::select("label", sample_info$label) %>%
  data.frame(row.names = 1, check.names = F) %>%
  filter(if_any(where(is.numeric), ~ .x > 0)) %>%
  t() # Transpose: rows as samples, columns as variables

Y <- pull(sample_info, "class") %>% as.factor()

# --- Principal Component Analysis (PCA) ---

# Run PCA method on data
pca.tgy = pca(X, ncomp = 10, center = TRUE, scale = TRUE) 
pdf("pca_bar_plot.pdf")
plot(pca.tgy)  # Barplot of the eigenvalues (explained variance per component)
dev.off()

pdf("pca_score_plot.pdf")
plotIndiv(pca.tgy, group = Y, ind.names = FALSE, # Plot the samples projected
          legend = TRUE, title = 'PCA comp 1 - 2') # onto the PCA subspace
dev.off()

# --- initial sPLS-DA Model ---

srbct.splsda <- splsda(X, Y, ncomp = 10)

# Plot the samples projected onto the first two components of the PLS-DA subspace
pdf("splsda_score_plot.pdf")
plotIndiv(srbct.splsda , comp = 1:2, 
          group = Y, ind.names = FALSE,  # Color points by class
          ellipse = TRUE, # Include 95% confidence ellipse for each class
          legend = TRUE, title = '(a) PLSDA with confidence ellipses')
dev.off()

# Use the max.dist measure to form decision boundaries between classes
background = background.predict(srbct.splsda, comp.predicted=2, dist = "max.dist")

# Plot the samples with prediction background
pdf("splsda_score_plot_bg.pdf")
plotIndiv(srbct.splsda, comp = 1:2,
          group = Y, ind.names = FALSE, 
          background = background, # Include prediction background for each class
          legend = TRUE, title = " (b) PLSDA with prediction background")
dev.off()

# --- Model Tuning and Performance Evaluation ---

# Undergo performance evaluation to tune the number of components
perf.splsda.tgy <- perf(srbct.splsda, validation = "Mfold", 
                          folds = 5, nrepeat = 10, cpus = 10, # Use repeated cross-validation
                          progressBar = FALSE, auc = TRUE) # Include AUC values

# Plot performance evaluation outcome across all components
pdf("splsda_performance.pdf")
plot(perf.splsda.tgy, col = color.mixo(5:7), sd = TRUE,
     legend.position = "horizontal")
dev.off()

# Optimal number of components according to perf()
perf.splsda.tgy$choice.ncomp 

# Grid of possible keepX values (number of variables) to test
list.keepX <- c(1:10,  seq(20, 300, 10))

# Tuning process to determine the optimal number of variables per component
tune.splsda.tgy <- tune.splsda(X, Y, ncomp = 8, 
                                 validation = 'Mfold',
                                 folds = 5, nrepeat = 10, 
                                 dist = 'max.dist', 
                                 measure = "BER", # Use Balanced Error Rate
                                 test.keepX = list.keepX,
                                 cpus = 10) # Parallelization to decrease runtime

pdf("splsda_tuning_variable.pdf")
plot(tune.splsda.tgy, col = color.jet(8)) # Plot output of variable selection tuning
dev.off()

# Optimal values according to tune.splsda()
tune.splsda.tgy$choice.ncomp$ncomp 
tune.splsda.tgy$choice.keepX 

optimal.ncomp <- tune.splsda.tgy$choice.ncomp$ncomp
optimal.keepX <- tune.splsda.tgy$choice.keepX[1:optimal.ncomp]

# --- Final sPLS-DA Model ---

final.splsda <- splsda(X, Y, 
                       ncomp = optimal.ncomp, 
                       keepX = optimal.keepX)

pdf("splsda_score_plot_final.pdf")
plotIndiv(final.splsda, comp = c(1,2), 
          group = Y, ind.names = FALSE, 
          ellipse = TRUE, legend = TRUE, 
          title = ' (a) sPLS-DA Final Model, comp 1 & 2')

plotIndiv(final.splsda, comp = c(1,3), 
          group = Y, ind.names = FALSE,  
          ellipse = TRUE, legend = TRUE, 
          title = '(b) sPLS-DA Final Model, comp 1 & 3')
dev.off()

# Set homogeneous styling for the legend
legend=list(legend = levels(Y), 
            col = unique(color.mixo(Y)), 
            title = "Product Type", 
            cex = 0.7)

# Generate Clustered Image Map (Heatmap)
pdf("splsda_heatmap.pdf")
cim <- cim(final.splsda, row.sideColors = color.mixo(Y), scale = TRUE,
            legend = legend)
dev.off()

# --- Prediction and Validation ---

# Train model on training set (Note: Ensure X.train and Y.train are defined)
train.splsda.tgy <- splsda(X.train, Y.train, ncomp = optimal.ncomp, keepX = optimal.keepX)

# Apply model to the test set
predict.splsda.tgy <- predict(train.splsda.tgy, X.test, 
                                 dist = "mahalanobis.dist")

# Evaluate results for Component 2
predict.comp2 <- predict.splsda.tgy$class$mahalanobis.dist[,2]
table(factor(predict.comp2, levels = levels(Y)), Y.test)

# Evaluate results for Component 8
predict.comp3 <- predict.splsda.tgy$class$mahalanobis.dist[,8]
cm_table <- table(factor(predict.comp3, levels = levels(Y)), Y.test) 

# Convert Confusion Matrix to long-format dataframe for ggplot
cm_df <- as.data.frame(as.table(cm_table))
names(cm_df) <- c("Predicted", "Actual", "Frequency")

# Create Heatmap of the Confusion Matrix
ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Frequency)) +
  geom_tile() +
  geom_text(aes(label = Frequency), color = "black", size = 4) +
  scale_fill_gradient2(low = "#FFFFFF", mid = '#59A2CF', high = "#08316D", midpoint = 8) +
  theme_minimal() +
  labs(title = "Confusion Matrix",
       x = "Actual Class",
       y = "Predicted Class") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("splsda_confusion_matrix.pdf", width = 10, height = 10)

# Generate ROC curve and calculate AUC
auc.splsda = auroc(train.splsda.tgy, roc.comp = 8, print = FALSE)
ggsave("splsda_model_roc_auc.pdf")