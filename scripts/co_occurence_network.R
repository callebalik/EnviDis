# Load necessary libraries
library(igraph)
library(readr)
library(scales)

# Read the CSV file
co_occurrence_matrix <- read_csv("scripts/co_occurrence_matrix.csv", show_col_types = FALSE)

# Convert the data frame to a matrix
co_occurrence_matrix <- as.matrix(co_occurrence_matrix[,-1])

# Create a graph from the adjacency matrix
g <- graph_from_adjacency_matrix(co_occurrence_matrix, mode = "undirected", weighted = TRUE, diag = FALSE)

# Set seed for graph plot
set.seed(1)

# Identification of all nodes with less than X edges
# verticesToRemove <- V(g)[degree(g) < 4]
# These edges are removed from the graph
# g <- delete_vertices(g, verticesToRemove)

# Assign colors to nodes (search term blue, others orange)
coocTerm <- ""  # Define the search term
V(g)$color <- ifelse(V(g)$name == coocTerm, 'cornflowerblue', 'orange')

# Set edge colors
E(g)$color <- adjustcolor("DarkGray", alpha.f = .5)
# Scale significance between 1 and 10 for edge width
E(g)$width <- scales::rescale(E(g)$weight, to = c(1, 10))

# Set edges with radius
E(g)$curved <- 0.15
# Size the nodes by their degree of networking (scaled between 5 and 15)
V(g)$size <- scales::rescale(log(degree(g)), to = c(5, 15))

# Define the frame and spacing for the plot
par(mai=c(0,0,1,0))

# Final Plot
plot(
  g,
  layout = layout.fruchterman.reingold, # Force Directed Layout
  main = paste(coocTerm, ' Graph'),
  vertex.label.family = "sans",
  vertex.label.cex = 0.8,
  vertex.shape = "circle",
  vertex.label.dist = 0.5,          # Labels of the nodes moved slightly
  vertex.frame.color = adjustcolor("darkgray", alpha.f = .5),
  vertex.label.color = 'black',     # Color of node names
  vertex.label.font = 2,            # Font of node names
  vertex.label = V(g)$name,         # Node names
  vertex.label.cex = 1              # Font size of node names
)
