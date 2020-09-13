library(rtracklayer)
library(BSgenome.Scerevisiae.UCSC.sacCer3)
data_dir = 'Documents/Projects/Nucleosomes/code/ChromWave_python3/data/hu2014/'
files = list.files(data_dir)
files = files[grep('bed', files)]
files = files[-grep('gz',files)]

for(filename in files){
  TF = import.bed(paste0(data_dir,filename))
  
  seqlevels(TF)[order(seqlevels(TF))] ==  seqlevels(Scerevisiae)[order( seqlevels(Scerevisiae))]
  seqlevels(TF) = seqlevels(TF)[order(seqlevels(TF))]
  
  seqlevels(TF)=  seqlevels(Scerevisiae)
  
  seqinfo(TF) = seqinfo(Scerevisiae)
  
  
  unroll = slidingWindows(TF, width = 1, step=1)
  unroll = unlist(unroll)
  
  hits = findOverlaps(unroll, TF)
  unroll$score = NA
  unroll[queryHits(hits)]$score = TF[subjectHits(hits)]$score
  
  newfilename = paste0(strsplit(filename,'\\.')[[1]][[1]],'_unrolled.bed')
  export.bed(unroll, paste0(data_dir,newfilename))

}



