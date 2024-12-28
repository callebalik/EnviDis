rm(list = ls())

# Libraries ---------------------------------------------------------------
library(visNetwork)
library(igraph)
library(tidyverse)
library(shiny)
library(readr)

# File Paths --------------------------------------------------------------
co_occurrence_matrix_file <- "tests/data/generated/co_occurrence_matrix.csv"
exported_network_dir <- "tests/data/export/network"

# Create the directory if it does not exist
if (!dir.exists(exported_network_dir)) {
  dir.create(exported_network_dir, recursive = TRUE)
}

node_positions_file <- file.path(exported_network_dir, "node_positions.csv")
network_file <- file.path(exported_network_dir, "network.html")

# Clense previous data
# if (file.exists(network_file)) {
#   file.remove(network_file)
# }

# Data Preparation --------------------------------------------------------

# Read the co-occurrence matrix from the CSV file
co_occurrence_matrix <- read.csv(co_occurrence_matrix_file, row.names = 1)

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

# Convert nodes and edges to an igraph object
graph <- graph_from_data_frame(edges, directed = FALSE)

# Louvain Comunity Detection
cluster <- cluster_louvain(graph)

cluster_df <- data.frame(as.list(membership(cluster)))
cluster_df <- as.data.frame(t(cluster_df))
cluster_df$label <- rownames(cluster_df)

# Create group column
nodes <- left_join(nodes, cluster_df, by = "label")
colnames(nodes)[3] <- "group"

# Ensure 'from' and 'to' columns exist in edges data frame
if (!all(c("from", "to") %in% colnames(edges))) {
  stop("The 'from' and 'to' columns are missing in the edges data frame.")
}

# Ensure 'id' column is unique in nodes data frame
nodes <- nodes %>% distinct(id, .keep_all = TRUE)

# Create the network visualization
network <- visNetwork(nodes, edges, width = "100%", height = "1000px") %>%
  # fix randomSeed to keep positions consistent
  visIgraphLayout(physics = TRUE, smooth = TRUE, randomSeed = 123, layout = "layout_nicely") %>% # This will use the igraph layout that more clearly shows the clusters
  visNodes(
    shape = "circle",
    color = list(
      background = "#67d7b2ad",
      border = "#0d3d48",
      highlight = "#ffd760"
    ),
    shadow = list(enabled = TRUE, size = 5),
    font = list(
      color = "#000000", # Font color
      size = 13, # Font size
      face = "calibri" # Font family
    )
  ) %>%
  visEdges(
    shadow = FALSE,
    color = list(color = "#36a0c1d8", highlight = "#fe7e7e"),
    smooth = list(enabled = TRUE, type = "continuous"),
  ) %>%
  visOptions(
    highlightNearest = list(enabled = TRUE, degree = 1, hover = TRUE), # nolint: indentation_linter.
    selectedBy = "group",
    nodesIdSelection = TRUE,
  ) %>%
  visLayout(randomSeed = 14) %>%
  visPhysics(
    # choose the solver (‘barnesHut’, ‘repulsion’, ‘hierarchicalRepulsion’, ‘forceAtlas2Based’), and set options
    solver = "forceAtlas2Based",
    stabilization = TRUE
  )

# Save the network visualization to an HTML file
visSave(network, file = network_file, selfcontained = FALSE)
