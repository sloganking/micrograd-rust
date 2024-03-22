// Import Value type from main.rs
use crate::Value;

use uuid::Uuid;

use petgraph::dot::Dot;
use petgraph::prelude::{DiGraph, NodeIndex};
use std::collections::HashMap;

use std::fs::File;
use std::io::Write;

fn value_to_graph_recursive(
    value: &Value,
    graph: &mut DiGraph<String, String>,
    node_map: &mut HashMap<Uuid, NodeIndex>,
) -> NodeIndex {
    let uuid = value.borrow().uuid;

    if let Some(&node_index) = node_map.get(&uuid) {
        return node_index;
    }

    let node_index = graph.add_node(format!(
        "data={:.1} grad={:.1}",
        value.borrow().data,
        value.borrow().grad
    ));
    node_map.insert(uuid, node_index);

    for prev_value in value.borrow().prev.iter() {
        let prev_node_index = value_to_graph_recursive(prev_value, graph, node_map);
        graph.add_edge(
            prev_node_index,
            node_index,
            value.borrow().op.clone().unwrap_or_default(),
        );
    }

    node_index
}

fn value_to_graph(value: &Value) -> DiGraph<String, String> {
    let mut graph = DiGraph::new();
    let mut node_map = HashMap::new();

    value_to_graph_recursive(value, &mut graph, &mut node_map);

    graph
}

pub fn create_graphviz(g: &Value, filename: &str) {
    g.backward();
    let graph = value_to_graph(&g);
    let mut dot = format!("{:?}", Dot::new(&graph));

    // Hacky way to adjust graphviz output
    dot = dot.replace("\\\"", "");
    dot.insert_str(10, "    rankdir=\"LR\"");
    dot.insert_str(10, "    node [shape=box]\n");
    println!("{}", dot);

    let mut file = File::create(filename).unwrap();
    file.write_all(dot.as_bytes()).unwrap();
}