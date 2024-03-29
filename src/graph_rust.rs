// this file is a different rendering implemenation than graph.rs.
// This wil will use the graphviz-rust lib instead.

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
    let graph = graph!(id!("test"), nodes);

    graph
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
