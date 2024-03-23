mod graph;
mod value;
use crate::value::Value;

fn main() {
    let a = Value::from(3.0);
    let b = Value::from(4.0);
    let c = Value::from(5.0);
    let d = (a - b) * c;

    d.backward();

    println!("{:#?}", *d);

    if graph::render_graph(&d).is_none() {
        println!("Error: Rendering dot file to an image failed. Please ensure that you have graphviz installed (https://graphviz.org/download/), and that the \"dot\" command is runnable from your terminal.");
    }
}
