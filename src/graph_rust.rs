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
    let mut all_values = get_prevs_of_recursive(v, &mut set);
    all_values.push(v.clone());
    all_values
}

fn value_to_statements(
    v: &Value,
    values_corresponding_op_node: &mut HashMap<Uuid, u128>,
) -> Vec<Stmt> {
    let mut statements = vec![];

    let id = &v.borrow().uuid.as_u128();
    let label = format!(
        "\"data={:.4} grad={:.4}\"",
        v.borrow().data,
        v.borrow().grad,
    );
    let node = stmt!(node!(id; NodeAttributes::shape(shape::box_),  NodeAttributes::label(label)));
    statements.push(node);

    if let Some(op) = &v.borrow().op {
        let label_node_id = Uuid::new_v4().as_u128();
        values_corresponding_op_node.insert(v.borrow().uuid, label_node_id);
        let label_node = stmt!(
            node!(label_node_id; NodeAttributes::shape(shape::oval),  NodeAttributes::label("\"".to_string() + op + "\""))
        );
        statements.push(label_node);

        let op_edge = stmt!(edge!(node_id!(label_node_id) => node_id!(id)));
        statements.push(op_edge);
    }

    statements
}

fn create_graph(v: &Value) -> Graph {
    // let graph = graph!(directed);

    let mut values_corresponding_op_node: HashMap<Uuid, u128> = HashMap::new();

    let values = get_all_values(v);

    let subgraph_map = get_subgraph_map(v);

    let mut graph_statements = vec![];

    // create all nodes in all subgraphs
    let subgraphs = {
        let mut subgraphs = vec![];
        for (subgraph_id, subgraph_values) in subgraph_map.iter() {
            let mut subgraph_statements = vec![];

            // add all the nodes to the subgraph
            for value in subgraph_values.iter() {
                subgraph_statements.extend(value_to_statements(
                    value,
                    &mut values_corresponding_op_node,
                ));
            }

            // add attributes to the subgraph
            let attributes = vec![
                attr!("label", "\"Neuron\""),
                SubgraphAttributes::color(color_name::blue),
                // SubgraphAttributes::bgcolor(color_name::red),
            ];
            let attributes_statements: Vec<Stmt> = attributes.into_iter().map(Stmt::from).collect();

            subgraph_statements.extend(attributes_statements);

            let subgraph_id = "cluster".to_owned() + subgraph_id.as_u128().to_string().as_str();
            subgraphs.push(stmt!(subgraph!(subgraph_id, subgraph_statements)));
        }
        subgraphs
    };

    graph_statements.extend(subgraphs);

    // create all nodes outside of subgraphs
    let nodes_outside_subgraphs = {
        let mut nodes_outside_subgraphs = vec![];
        for value in values.iter() {
            if value.borrow().subgraph_id.is_some() {
                continue;
            }

            nodes_outside_subgraphs.extend(value_to_statements(
                value,
                &mut values_corresponding_op_node,
            ));
        }
        nodes_outside_subgraphs
    };
    graph_statements.extend(nodes_outside_subgraphs);

    // create all edges
    let mut edge_statements = vec![];
    for value in values.iter() {
        let node_or_label_node_id = match values_corresponding_op_node.get(&value.borrow().uuid) {
            Some(op_node_id) => *op_node_id,
            None => value.borrow().uuid.as_u128(),
        };
        for prev in value.borrow().prev.iter() {
            let edge = stmt!(
                edge!(node_id!(prev.borrow().uuid.as_u128()) => node_id!(node_or_label_node_id))
            );
            edge_statements.push(edge);
        }
    }

    graph_statements.extend(edge_statements);

    // create graph attributes
    let graph_attributes = vec![attr!("rankdir", "LR")];
    let attributes_statements: Vec<Stmt> = graph_attributes.into_iter().map(Stmt::from).collect();

    graph_statements.extend(attributes_statements);

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
