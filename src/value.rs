use std::collections::HashSet;
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::hash::Hasher;
use std::iter::Sum;
use std::ops::{self, Add, Div, Mul, Sub};
use std::{cell::RefCell, rc::Rc};
use uuid::Uuid;

#[derive(Debug)]
pub struct ValueData {
    pub data: f64,
    pub grad: f64,
    pub backward: Option<fn(value: &ValueData)>,
    pub prev: Vec<Value>,
    pub op: Option<String>,
    pub uuid: Uuid,
}

impl ValueData {
    fn new(data: f64) -> ValueData {
        ValueData {
            data,
            grad: 0.0,
            backward: None,
            prev: Vec::new(),
            op: None,
            uuid: Uuid::new_v4(),
        }
    }
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

// Lets us do `value.borrow().data` instead of `value.0.borrow().data`
impl ops::Deref for Value {
    type Target = Rc<RefCell<ValueData>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().uuid == other.borrow().uuid
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.borrow().uuid.hash(state);
    }
}

impl Value {
    fn new(value: ValueData) -> Value {
        Value(Rc::new(RefCell::new(value)))
    }

    pub fn backward(&self) {
        let mut topo = self.build_topo();
        topo.reverse();

        self.borrow_mut().grad = 1.0;
        for v in topo {
            if let Some(backprop) = v.borrow().backward {
                backprop(&v.borrow());
            }
        }
    }

    fn build_topo(&self) -> Vec<Value> {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<Value> = HashSet::new();
        self._build_topo(&mut topo, &mut visited);
        topo
    }

    fn _build_topo(&self, topo: &mut Vec<Value>, visited: &mut HashSet<Value>) {
        if visited.insert(self.clone()) {
            self.borrow().prev.iter().for_each(|child| {
                child._build_topo(topo, visited);
            });
            topo.push(self.clone());
        }
    }

    pub fn relu(&self) -> Self {
        let mut new_value = ValueData::new(self.borrow().data.max(0.0));

        new_value.prev = vec![self.clone()];
        new_value.op = Some(String::from("ReLU"));
        new_value.backward = Some(|value: &ValueData| {
            if value.data > 0.0 {
                value.prev[0].borrow_mut().grad += value.grad;
            }
        });

        Value::new(new_value)
    }

    pub fn pow(&self, n: Value) -> Self {
        let mut new_value = ValueData::new(self.borrow().data.powf(n.borrow().data));

        new_value.prev = vec![self.clone(), n];
        new_value.op = Some(format!("pow()"));
        new_value.backward = Some(|value: &ValueData| {
            let base = value.prev[0].borrow().data;
            let exp = value.prev[1].borrow().data;
            value.prev[0].borrow_mut().grad += value.grad * exp * base.powf(exp - 1.0);
        });

        Value::new(new_value)
    }

    pub fn tanh(&self) -> Self {
        let mut new_value = ValueData::new(self.borrow().data.tanh());

        new_value.prev = vec![self.clone()];
        new_value.op = Some(format!("tanh()"));
        new_value.backward = Some(|value: &ValueData| {
            let tanh = value.data.tanh();
            value.prev[0].borrow_mut().grad += value.grad * (1.0 - tanh * tanh);
        });

        Value::new(new_value)
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

        new_value.prev = vec![self, other];
        new_value.op = Some(String::from("+"));
        new_value.backward = Some(|value: &ValueData| {
            value.prev[0].borrow_mut().grad += value.grad;
            value.prev[1].borrow_mut().grad += value.grad;
        });

        Value::new(new_value)
    }
}

impl Sub for Value {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut new_value = ValueData::new(self.borrow().data - other.borrow().data);

        new_value.prev = vec![self, other];
        new_value.op = Some(String::from("-"));
        new_value.backward = Some(|value: &ValueData| {
            value.prev[0].borrow_mut().grad += value.grad;
            value.prev[1].borrow_mut().grad -= value.grad;
        });

        Value::new(new_value)
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut new_value = ValueData::new(self.borrow().data * other.borrow().data);

        new_value.prev = vec![self, other];
        new_value.op = Some(String::from("*"));
        new_value.backward = Some(|value: &ValueData| {
            value.prev[0].borrow_mut().grad += value.grad * value.prev[1].borrow().data;
            value.prev[1].borrow_mut().grad += value.grad * value.prev[0].borrow().data;
        });

        Value::new(new_value)
    }
}

impl Div for Value {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let mut new_value = ValueData::new(self.borrow().data / other.borrow().data);

        new_value.prev = vec![self, other];
        new_value.op = Some(String::from("/"));
        new_value.backward = Some(|value: &ValueData| {
            value.prev[0].borrow_mut().grad += value.grad / value.prev[1].borrow().data;
            value.prev[1].borrow_mut().grad +=
                value.grad * value.prev[0].borrow().data / value.prev[1].borrow().data.powi(2);
        });

        Value::new(new_value)
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut new_value = ValueData::new(0.0);

        let sum: f64 = iter
            .map(|val| {
                new_value.prev.push(val.clone());
                val.borrow().data
            })
            .sum();
        new_value.data = sum;

        new_value.op = Some(String::from("+"));
        new_value.backward = Some(|value: &ValueData| {
            for val in value.prev.iter() {
                val.borrow_mut().grad += value.grad;
            }
        });

        Value::new(new_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_grad() {
        let a = Value::from(3.0);
        assert_eq!(a.borrow().grad, 0.0);
        a.backward();
        assert_eq!(a.borrow().grad, 1.0);
    }

    #[test]
    fn add() {
        let a = Value::from(3.0);
        let b = Value::from(4.0);
        let c = a.clone() + b.clone();

        c.backward();

        assert_eq!(c.borrow().data, 7.0);
        assert_eq!(a.borrow().grad, 1.0);
        assert_eq!(b.borrow().grad, 1.0);
    }

    #[test]
    fn add_self() {
        let a = Value::from(3.0);
        let b = a.clone() + a.clone();

        b.backward();

        assert_eq!(b.borrow().data, 6.0);
        assert_eq!(a.borrow().grad, 2.0);
    }

    #[test]
    fn multiply() {
        let a = Value::from(3.0);
        let b = Value::from(4.0);
        let c = a.clone() * b.clone();

        c.backward();

        assert_eq!(c.borrow().data, 12.0);
        assert_eq!(a.borrow().grad, 4.0);
        assert_eq!(b.borrow().grad, 3.0);
    }

    // #[test]
    // fn multiply_self() {
    //     let a = Value::from(3.0);
    //     let b = a.clone() * a.clone();

    //     b.backward();

    //     assert_eq!(b.borrow().data, 9.0);
    //     assert_eq!(a.borrow().grad, 6.0);
    // }

    #[test]
    fn subtract() {
        let a = Value::from(3.0);
        let b = Value::from(4.0);
        let c = a.clone() - b.clone();

        c.backward();

        assert_eq!(c.borrow().data, -1.0);
        assert_eq!(a.borrow().grad, 1.0);
        assert_eq!(b.borrow().grad, -1.0);
    }

    #[test]
    fn subtract_self() {
        let a = Value::from(3.0);
        let b = a.clone() - a.clone();

        b.backward();

        assert_eq!(b.borrow().data, 0.0);
        assert_eq!(a.borrow().grad, 0.0);
    }

    #[test]
    fn relu() {
        // positive input
        {
            let a = Value::from(3.0);
            let b = a.relu();

            b.backward();

            assert_eq!(b.borrow().data, 3.0);
            assert_eq!(a.borrow().grad, 1.0);
        }

        // negative input
        {
            let a = Value::from(-3.0);
            let b = a.relu();

            b.backward();

            assert_eq!(b.borrow().data, 0.0);
            assert_eq!(a.borrow().grad, 0.0);
        }
    }
}
