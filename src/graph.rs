// Import Value type from main.rs
use crate::Value;
use petgraph::dot::Dot;
use petgraph::prelude::{DiGraph, NodeIndex};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use uuid::Uuid;

fn value_to_graph_recursive(
    value: &Value,
    graph: &mut DiGraph<String, String>,
    node_map: &mut HashMap<Uuid, NodeIndex>,
) -> NodeIndex {
    let uuid = value.borrow().uuid;

    if let Some(&node_index) = node_map.get(&uuid) {
        return node_index;
    }

    let data_node_index = graph.add_node(format!(
        "data={:.4} grad={:.4}",
        value.borrow().data,
        value.borrow().grad
    ));
    node_map.insert(uuid, data_node_index);

    let node_index_that_prevs_point_to = if let Some(op) = &value.borrow().op {
        // add op label node
        let label_node_index = graph.add_node(format!("{}{}", op, "@label"));

        // add edge from label to data node
        graph.add_edge(label_node_index, data_node_index, "".to_owned());

        label_node_index
    } else {
        data_node_index
    };

    for prev_value in value.borrow().prev.iter() {
        let prev_data_node_index = value_to_graph_recursive(prev_value, graph, node_map);
        graph.add_edge(
            prev_data_node_index,
            node_index_that_prevs_point_to,
            "".to_owned(),
        );
    }

    data_node_index
}

fn value_to_graph(value: &Value) -> DiGraph<String, String> {
    let mut graph = DiGraph::new();
    let mut node_map = HashMap::new();

    value_to_graph_recursive(value, &mut graph, &mut node_map);

    graph
}

pub fn create_graphviz(g: &Value, filename: &str) {
    let graph = value_to_graph(&g);
    let mut dot = format!("{:?}", Dot::new(&graph));

    // Hacky way to adjust graphviz output
    dot = dot.replace("\\\"", "");
    dot.insert_str(10, "    rankdir=\"LR\"");

    let default_node_shape = "box";
    dot.insert_str(10, &format!("    node [shape={}]\n", default_node_shape));

    // turn label nodes into ovals
    let dot = {
        let mut new_dot = String::new();
        for line in dot.lines() {
            if line.contains("@label") {
                new_dot.push_str("    node [shape=oval]");
                new_dot.push('\n');
                new_dot.push_str(&line.replace("@label", ""));
                new_dot.push('\n');
                new_dot.push_str(&format!("    node [shape={}]", default_node_shape));
                new_dot.push('\n');
            } else {
                new_dot.push_str(line);
                new_dot.push('\n');
            }
        }
        new_dot
    };

    println!("{}", dot);

    let mut file = File::create(filename).unwrap();
    file.write_all(dot.as_bytes()).unwrap();
}

pub fn render_graph(v: &Value) {
    let dir = "test.dot";
    create_graphviz(&v, dir);

    // can be "png" or "svg".
    let output_file_type = "png";

    if std::process::Command::new("dot")
        .arg(format!("-T{}", output_file_type))
        .arg("-o")
        .arg(format!("graph.{}", output_file_type))
        .arg(dir)
        .output()
        .is_err()
    {
        println!("Error: Rendering dot file to an image failed. Please ensure that you have graphviz installed (https://graphviz.org/download/), and that the \"dot\" command is runnable from your terminal.");
    }
}
