rm(list = ls())

# Libraries ---------------------------------------------------------------
library(visNetwork)
library(igraph)
library(tidyverse)
library(shiny)
library(readr)

# Data Preparation --------------------------------------------------------

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

# Ensure 'group' column exists in node data frame
if (!"group" %in% colnames(nodes)) {
    nodes$group <- NA # or assign appropriate default values
}

# Ensure 'id' column exists in nodes data frame
if (!"id" %in% colnames(nodes)) {
    stop("The 'id' column is missing in the nodes data frame.")
}

# Ensure 'from' and 'to' columns exist in edges data frame
if (!all(c("from", "to") %in% colnames(edges))) {
    stop("The 'from' and 'to' columns are missing in the edges data frame.")
}

# Ensure 'id' column is unique in nodes data frame
nodes <- nodes %>% distinct(id, .keep_all = TRUE)

# Load saved node positions if they exist
positions_file <- "node_positions.csv"
if (file.exists(positions_file)) {
    node_positions <- read.csv(positions_file)
    if (!"id" %in% colnames(node_positions)) {
        stop("The 'id' column is missing in the node positions file.")
    }
    nodes <- merge(nodes, node_positions, by = "id", all.x = TRUE)
} else {
    node_positions <- data.frame(id = character(), x = numeric(), y = numeric(), stringsAsFactors = FALSE)
}

# Create the network visualization
network <- visNetwork(nodes, edges, width = "100%") %>%

    visNodes(
        shape = "circle",
        color = list(
            background = "#4eecef",
            border = "#0d3d48",
            highlight = "#FF8000"
        ),
        shadow = list(enabled = TRUE, size = 5),
        font = list(
            color = "#000000", # Font color
            size = 14, # Font size
            face = "calibri" # Font family
        )
    ) %>%
    visEdges(
        shadow = FALSE,
        color = list(color = "#0085AF", highlight = "#e6db0a")
    ) %>%
    visOptions(
        highlightNearest = list(enabled = TRUE, degree = 1, hover = TRUE), # nolint: indentation_linter.
        selectedBy = "group",
        nodesIdSelection = TRUE,
    ) %>%
    visLayout(randomSeed = 1) %>%
    visPhysics(stabilization = TRUE) %>%
    visEvents(stabilized = "function() {
    this.getPositions(function(positions) {
        var positionsData = [];
        for (var id in positions) {
        positionsData.push({id: id, x: positions[id].x, y: positions[id].y});
        }
        Shiny.onInputChange('node_positions', positionsData);
    });
    }")

# Save the network visualization to an HTML file
visSave(network, file = "tests/network.html", selfcontained = FALSE)

# Save node positions after stabilization
observeEvent(input$node_positions, {
    new_positions <- do.call(rbind, lapply(input$node_positions, as.data.frame))
    if (nrow(new_positions) > 0) {
        write.csv(new_positions, positions_file, row.names = FALSE)
        node_positions <<- new_positions
    }
})

# Assuming node positions are stored in a data frame called 'node_positions'
write.csv(node_positions, file = "node_positions.csv", row.names = FALSE)
