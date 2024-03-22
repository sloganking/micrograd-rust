use std::fmt::{self, Debug};
use std::iter::Sum;
use std::ops::{self, Add};
use std::{cell::RefCell, rc::Rc};

fn main() {
    println!("Hello, world!");

    pub struct ValueData {
        data: f64,
        grad: f64,
        backward: Option<fn(value: &ValueData)>,
        prev: Vec<Value>,
    }

    impl ValueData {
        fn new(data: f64) -> ValueData {
            ValueData {
                data,
                grad: 0.0,
                backward: None,
                prev: Vec::new(),
                // uuid: Uuid::new_v4(),
                // _backward: None,
                // _prev: Vec::new(),
                // _op: None,
            }
        }
    }

    pub struct Value(Rc<RefCell<ValueData>>);

    // Lets us do `value.borrow().data` instead of `value.0.borrow().data`
    impl ops::Deref for Value {
        type Target = Rc<RefCell<ValueData>>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl Value {
        fn new(value: ValueData) -> Value {
            Value(Rc::new(RefCell::new(value)))
        }
    }

    impl<T: Into<f64>> From<T> for Value {
        fn from(t: T) -> Value {
            Value::new(ValueData::new(t.into()))
        }
    }

    impl Debug for Value {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let v = &self.borrow();
            write!(f, "data={} grad={}", v.data, v.grad)
        }
    }

    impl Add for Value {
        type Output = Self;

        fn add(self, other: Self) -> Self {
            let mut new_value = ValueData::new(self.borrow().data + other.borrow().data);

            // new_value.backward = Some(|value: &ValueData| {
            //     value.prev[0].backward(value.grad);
            //     value.prev[1].backward(value.grad);
            // });
            new_value.prev.push(self);
            new_value.prev.push(other);
            Value::new(new_value)
        }
    }

    // impl Sum for Value {
    //     fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
    //         let first = iter.next().expect("must contain at least one Value");
    //         iter.fold(first, |acc, val| acc + val)
    //     }
    // }

    let a = Value::from(3.0);
    let b = Value::from(4.0);
    let c = a + b;
    println!("{:?}", c);
}
