rm(list = ls())

# Libraries ---------------------------------------------------------------
library(visNetwork)
library(igraph)
library(tidyverse)
library(shiny)

# Data Preparation --------------------------------------------------------

# Read the co-occurrence matrix
# co_occurrence_matrix <- read_csv("scripts/co_occurrence_matrix.csv", show_col_types = FALSE)
co_occurrence_matrix <- read.csv("tests/co_occurrence_matrix.csv", row.names = 1)

# Convert the matrix to long format
co_occurrence_long <- co_occurrence_matrix %>%
    rownames_to_column(var = "source") %>%
    tidyr::gather(key = "target", value = "weight", -source) %>%
    dplyr::filter(weight > 0)

# Create nodes and edges for visNetwork
nodes <- data.frame(id = unique(c(co_occurrence_long$source, co_occurrence_long$target)))
edges <- data.frame(from = co_occurrence_long$source, to = co_occurrence_long$target, value = co_occurrence_long$weight, label = co_occurrence_long$weight)

# Check if pandoc is installed
# if (Sys.which("pandoc") == "") {
#   stop("Pandoc is not installed. Please install pandoc from https://bookdown.org/yihui/rmarkdown-cookbook/install-pandoc.html")
# }

# Ensure 'group' column exists in node data frame
if (!"group" %in% colnames(nodes)) {
    nodes$group <- NA # or assign appropriate default values
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
