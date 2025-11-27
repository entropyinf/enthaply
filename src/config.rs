use crate::Res;

pub trait ConfigRefresher<C: PartialEq> {
    fn refresh(&mut self, old: &C, new: &C) -> Res<()>;
}
