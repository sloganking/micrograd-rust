mod graph;
mod value;
use crate::value::Value;

fn main() {
    let a = Value::from(3.0);
    let b = Value::from(4.0);
    let c = Value::from(5.0);
    let d = (a - b) * c;
    println!("{:#?}", *d);

    d.backward();

    println!("{:#?}", *d);

    graph::render_graph(&d);
}
