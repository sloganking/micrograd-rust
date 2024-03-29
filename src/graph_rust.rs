// this file is a different rendering implemenation than graph.rs.
// This wil will use the graphviz-rust lib instead.

use std::collections::HashMap;
use std::collections::HashSet;
use std::vec;

use crate::Value;
use graphviz_rust::dot_generator::*;
use graphviz_rust::dot_structures::*;
use graphviz_rust::{
    attributes::*,
    cmd::{CommandArg, Format},
    exec, exec_dot, parse,
    printer::{DotPrinter, PrinterContext},
};
use uuid::Uuid;

fn get_prevs_of_recursive(v: &Value, set: &mut HashSet<Value>) -> Vec<Value> {
    let mut values = vec![];
    for prev in v.borrow().prev.iter() {
        if set.contains(&prev) {
            continue;
        }
        set.insert(prev.clone());
        values.push(prev.clone());
        values.extend(get_prevs_of_recursive(prev, set));
    }
    values
}

fn get_all_values(v: &Value) -> Vec<Value> {
    let mut set = HashSet::new();
    get_prevs_of_recursive(v, &mut set)
}

fn create_graph(v: &Value) -> Graph {
    // let graph = graph!(directed);

    let values = get_all_values(v);

    let subgraph_map = get_subgraph_map(v);

    let mut graph_statements = vec![];

    let mut subgraphs = vec![];
    // create all nodes in subgraphs
    for (i, (subgraph_id, subgraph_values)) in subgraph_map.iter().enumerate() {
        let mut statements = vec![];

        // add all the nodes to the subgraph
        for value in subgraph_values.iter() {
            let id = &value.borrow().uuid.as_u128();

            let label = format!(
                "\"data={:.4} grad={:.4} {}\"",
                value.borrow().data,
                value.borrow().grad,
                value.borrow().op.as_ref().unwrap_or(&"".to_string())
            );

            let node = if i == 0 {
                stmt!(
                    node!(id; NodeAttributes::shape(shape::box_),  NodeAttributes::label(label), NodeAttributes::color(color_name::blue))
                )
            } else {
                stmt!(node!(id; NodeAttributes::shape(shape::box_),  NodeAttributes::label(label)))
            };
            statements.push(node);

            if let Some(op) = &value.borrow().op {
                let label_node_id = Uuid::new_v4().as_u128();
                let label_node = stmt!(
                    node!(label_node_id; NodeAttributes::shape(shape::oval),  NodeAttributes::label("\"".to_string() + op + "\""))
                );
                statements.push(label_node);

                let op_edge = stmt!(edge!(node_id!(label_node_id) => node_id!(id)));
                statements.push(op_edge);
            }
        }

        // add attributes to the subgraph
        let attributes = vec![
            attr!("label", "test"),
            SubgraphAttributes::color(color_name::blue),
            SubgraphAttributes::bgcolor(color_name::red),
        ];
        let attributes_statements: Vec<Stmt> = attributes.into_iter().map(Stmt::from).collect();

        statements.extend(attributes_statements);

        // let subgraph = vec![stmt!(subgraph!(subgraph_id.as_u128(), subgraph_nodes))];
        subgraphs.push(stmt!(subgraph!(subgraph_id.as_u128(), statements)));
    }

    graph_statements.extend(subgraphs);

    // create all nodes outside of subgraphs

    let nodes_outside_subgraphs = {
        let mut nodes_outside_subgraphs = vec![];
        for value in values.iter() {
            if value.borrow().subgraph_id.is_some() {
                continue;
            }

            let id = &value.borrow().uuid.as_u128();
            let label = format!(
                "\"data={:.4} grad={:.4} {}\"",
                value.borrow().data,
                value.borrow().grad,
                value.borrow().op.as_ref().unwrap_or(&"".to_string())
            );
            let node =
                stmt!(node!(id; NodeAttributes::shape(shape::box_),  NodeAttributes::label(label)));
            nodes_outside_subgraphs.push(node);

            if let Some(op) = &value.borrow().op {
                let label_node_id = Uuid::new_v4().as_u128();
                let label_node = stmt!(
                    node!(label_node_id; NodeAttributes::shape(shape::oval),  NodeAttributes::label("\"".to_string() + op + "\""))
                );
                nodes_outside_subgraphs.push(label_node);

                let op_edge = stmt!(edge!(node_id!(label_node_id) => node_id!(id)));
                nodes_outside_subgraphs.push(op_edge);
            }
        }
        nodes_outside_subgraphs
    };

    graph_statements.extend(nodes_outside_subgraphs);

    // create all nodes
    let mut nodes = vec![];
    for value in values.iter() {
        let id = &value.borrow().uuid.as_u128();
        println!("id: {}", id);
        let label = format!(
            "\"data={:.4} grad={:.4} {}\"",
            value.borrow().data,
            value.borrow().grad,
            value.borrow().op.as_ref().unwrap_or(&"".to_string())
        );
        let node =
            stmt!(node!(id; NodeAttributes::shape(shape::box_),  NodeAttributes::label(label)));
        nodes.push(node);
    }

    // let stmts: Vec<Stmt> = nodes.into_iter().map(Stmt::from).collect();

    let attributes = vec![attr!("rankdir", "LR")];
    let graph = graph!(di id!("test"), graph_statements);

    graph
}

fn get_subgraph_map(v: &Value) -> HashMap<Uuid, Vec<Value>> {
    let mut map: HashMap<Uuid, Vec<Value>> = HashMap::new();

    let values = get_all_values(v);

    for value in values.iter() {
        if let Some(subgraph_id) = value.borrow().subgraph_id {
            match map.get_mut(&subgraph_id) {
                Some(vec) => vec.push(value.clone()),
                None => {
                    map.insert(subgraph_id, vec![value.clone()]);
                }
            }
        }
    }

    map
}

pub fn render_graph(v: &Value) -> Option<()> {
    let dir = "test.dot";
    // can be "png" or "svg".
    let output_file_type = "png";

    let graph = create_graph(&v);

    let text = graph.print(&mut PrinterContext::default());

    println!("===== text: =====\n{}", text);

    // save to file
    println!("saving to file");
    exec(
        graph,
        &mut PrinterContext::default(),
        vec![
            Format::Png.into(),
            CommandArg::Output("./graph2.png".to_string()),
        ],
    )
    .unwrap();
    println!("saved to file");

    Some(())
}
