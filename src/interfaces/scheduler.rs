pub use crate::interfaces::scheduler_mixin::{SchedulerMixin, SchedulerStepOutput};

pub trait VideoScheduler: SchedulerMixin {}

impl<T: SchedulerMixin + ?Sized> VideoScheduler for T {}
