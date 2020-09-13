require(stringr)
require(data.table)
require(ggplot2)
require(reshape2)
require(tidyr)
require(seqLogo)
require(readr)
require(dplyr)
require(TFBSTools)

list.dirs <- function(path=".", pattern=NULL, full.names=TRUE) {
  # Lists the directories present within a path.
  # Credit: http://stackoverflow.com/questions/4749783
  #
  # Args:
  #      See list.files
  #
  # Returns:
  #      A vector of directories within the path being searched.
  all <- list.files(path, pattern, full.names=full.names, recursive=FALSE)
  all[file.info(all)$isdir]
}
# from tataR in case it's not installed

readMotif<-function (motif.csv){
    motif.name <- gsub(".csv", "", basename(motif.csv))
    pcm <- read_csv(motif.csv, col_names = F, skip = 1) %>% select(-X1) %>%
        as.matrix()
    rownames(pcm) <- c("A", "C", "G", "T")
    pfm <- PFMatrix(ID = motif.name, profileMatrix = pcm)
    return(pfm)
}

readMotifs<- function (motif.csvs){
    pfms <- lapply(motif.csvs, function(motif.csv) readMotif(motif.csv))
    names(pfms) <- gsub(".csv", "", basename(motif.csvs))
    pfms <- do.call(PFMatrixList, pfms)
    return(pfms)
}

plotAllMotifs <- function(motifs_directory, output_directory) {
  motif.csvs <- list.files(motifs_directory, pattern = 'csv', full.names = T)
  # if there are no motifs with sufficient information content...
  if(length(motif.csvs) == 0) {
    # ...then just don't do anything and get out of here
    return(FALSE)
  }
  motifs <- gsub('.csv', '', basename(motif.csvs))
  pfm.list <- readMotifs(motif.csvs)

  for(i in 1:length(pfm.list)) {
    motif_pwm <- toICM(pfm.list[[i]])
    pdf(paste0(output_directory, '/',paste0(motifs[i], '-pwm.pdf')),width=35,height=20)
    print(seqLogo(motif_pwm, ic.scale=TRUE, xaxis = FALSE, yaxis = FALSE))
    dev.off()
  }
}


args <- commandArgs(trailingOnly = TRUE)

plotAllMotifs(args[1], args[2])
