# ..................clusterung....................................
imdb<-read.csv('D:\\V data analytics\\Python\\Deepesh materials\\deepesh r documents-clustering\\data.txt',header=F,sep="|",quote = "\"",na.strings = c('',' ','NA'))
colSums(is.na(imdb))
View(imdb)
colnames(imdb)=c("ID","Title","ReleaseDate","VideoReleaseDate","IMDB","Unknown",
                 "Action","Adventure","Animation","Children's","Comedy","Crime",
                 "Documentary","Drama","Fantasy","FlimNoir","Horror","Musical","Mystery",
                 "Romance","Scifi","Thriller","War","Western")
imdb$ID=NULL
imdb$ReleaseDate=NULL
imdb$VideoReleaseDate=NULL
imdb$IMDB=NULL
imdb<-unique(imdb)
#............................Hirarchial clustering..........................................
View(imdb)
imdbdist<-dist(imdb[,-1],method ='euclidian')
#model fitting
m1=hclust(imdbdist,method = 'ward.D')
plot(m1,labels = imdb$Title)
rect.hclust(m1,k=10,border = 'red')
imdb.clusters<-cutree(m1,k=10)
library(cluster)
clusplot(imdb[,-1],imdb.clusters,main="2D representation of the Cluster Solution",color=T,shade=T,labels=2,lines=0)
#
table(imdb.clusters)
tapply(imdb$Action,imdb.clusters,mean)*100
tapply(imdb$Romance,imdb.clusters,mean)*100
tapply(imdb$`Children's`,imdb.clusters,mean)*100
tapply(imdb$Crime,imdb.clusters,mean)*100
#..........................................
imdb$clusters<-imdb.clusters
str(imdb)
View(imdb)
recommended_movies<-imdb[,c(1,21)]
View(recommended_movies)
# saving the movie list as csv
write.csv(recommended_movies,'D:\\V data analytics\\R\\Clystering of IMDB\\movies_clustered.csv',row.names = TRUE)
#........................KMeans clustering..........................................
set.seed(1)
m2<-kmeans(imdb[,-1],10)
#silhoutte plot
d=c()
for (x in 1:10)
{a=kmeans(imdb[,-1],x)
d[x]=a$tot.withinss}
plot(c(1:10),d,type="b",xlab="Number of clusters",ylab="within group sum of squares")
     table(m2$cluster)
#........................................Fuzzy FCM......................................
imdb_dist1<-dist(imdb[,-1])
m3<-fanny(as.matrix(imdb_dist1),k=10,maxit=2000)
names(m3)
m3$membership
table(m3$clustering)
library(factoextra)
fviz_silhouette(m3,label = TRUE)

