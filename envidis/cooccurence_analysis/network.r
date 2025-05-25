# Load necessary libraries
library(visNetwork)

# Read the co-occurrence matrix from the CSV file
co_occurrence_matrix <- read.csv("tests/co_occurrence_matrix.csv", row.names = 1)

# Calculate the degree of each node
node_degrees <- rowSums(co_occurrence_matrix) + colSums(co_occurrence_matrix)

# Calculate the number of unique connections for each node
unique_connections <- rowSums(co_occurrence_matrix > 0) + colSums(co_occurrence_matrix > 0)

# Create nodes data frame with size based on unique connections
nodes <- data.frame(id = colnames(co_occurrence_matrix), label = colnames(co_occurrence_matrix), value = unique_connections)

# Create edges data frame
edges <- data.frame()
for (i in seq_len(nrow(co_occurrence_matrix))) {
  for (j in i:ncol(co_occurrence_matrix)) {
    if (co_occurrence_matrix[i, j] > 0) {
      edges <- rbind(edges, data.frame(from = rownames(co_occurrence_matrix)[i], to = colnames(co_occurrence_matrix)[j], label = co_occurrence_matrix[i, j], value = co_occurrence_matrix[i, j]))
    }
  }
}

# Create the network
network <- visNetwork(nodes, edges) %>%
  visNodes(size = nodes$value) %>%
  visEdges(label = edges$label) %>%
  visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE)

# Save the network to a PNG file
visSave(network,
  file = "tests/2.html",
  selfcontained = FALSE)
